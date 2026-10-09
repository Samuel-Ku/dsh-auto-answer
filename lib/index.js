/**
 * dsh-auto-answer — host plugin that answers `ask_user_question` prompts with an LLM.
 *
 * The `ask_user_question` tool raises its popup through a different seam than
 * sandbox escalations: `ctx.userQuestions.ask(...)` runs a waterfall on
 * `user-questions/request` whose default answerer is the browser UI. This
 * plugin prepends a listener on that waterfall and answers option-only
 * questions itself; everything else falls through with `next()`, so the normal
 * popup is unaffected.
 *
 * Fail-closed by construction. The human popup is the fallback for every path
 * this plugin is not certain about:
 *   - any question without `options` (free text needs facts only the operator has)
 *   - no configured provider/model, missing llm service
 *   - a model verdict that is not `confident: true`
 *   - an unparseable reply, an unknown/duplicated option label, a wrong
 *     selection count for a single-select question, a missing question id —
 *     once `judge.repairAttempts` repair turns (default 1) have failed to fix it
 *   - timeout, abort, stream error, concurrency overflow
 *
 * A repairable failure is not immediately fatal: the reply is quoted back with
 * the reason it was rejected, and the model gets one more turn. The repair turns
 * share the request's single deadline, so they can never extend the wait.
 *
 * A "not confident" verdict is not final either: the `escalation.*` profile
 * (typically a bigger model given a much larger slice of the conversation) gets
 * a second opinion before the operator is bothered.
 *
 * And the operator is never waited on indefinitely: once the popup is open,
 * `fallback.afterMs` bounds the wait. On timeout the escalation profile is asked
 * to commit; failing that, the single "(Recommended)" option is taken. Only when
 * neither exists does the question keep waiting, and the audit says so.
 *
 * Context: the decision is made from the question text plus a truncated excerpt
 * of the session's own `user/message` / `assistant/message` history. Nothing
 * else is read, and nothing is written back into the session.
 *
 * Pure JavaScript (ESM); host code imports only node: builtins and same-package
 * peers (@deepseek-ai/cordis, @deepseek-ai/dsh-llm, @deepseek-ai/dsh-timeout).
 *
 * @module dsh-auto-answer
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { BlockAssembler, createUserMessage } from '@deepseek-ai/dsh-llm';
import { deadline } from '@deepseek-ai/dsh-timeout';

/** Cordis plugin name. */
export const name = 'dsh-auto-answer';

/**
 * Required services. Only `llm` is injected: the waterfall listener needs no
 * `userQuestions` handle (it is a plain event listener), and the plugin must
 * stay inert when no llm service is composed.
 */
export const inject = ['llm'];

/** Capability code used to distinguish our deadline from an upstream abort. */
const TIMEOUT_CODE = 'AUTO_ANSWER_TIMEOUT';

/** Fenced-JSON instruction appended to every prompt. */
const OUTPUT_CONTRACT = [
  'Reply with exactly this JSON (optionally inside a ``` fence):',
  '{"answers":[{"id":"<question id>","selected":["<option label>"]}],"confident":true,"reason":"..."}',
  'Write reason in Ukrainian, one short sentence.',
].join('\n');

/** Built-in system prompt (used when the config leaves `judge.systemPrompt` empty). */
export const DEFAULT_SYSTEM_PROMPT = [
  'You answer clarifying questions on behalf of the human operator of a DeepSeek Harness session.',
  'You are not the agent that asked, and you do not represent any agent\'s interests. Decide what the OPERATOR would most plausibly choose, using only the conversation excerpt and the question text you are given.',
  'Never invent facts about the operator. If the excerpt does not determine the answer, or the question asks for something only the operator knows (a path, a secret, a personal preference, a business decision), return {"confident": false} and nothing else.',
  'Select labels EXACTLY as they appear in that question\'s options - never paraphrase and never invent an option.',
  'Answer every question you were given. When a question does not set multiSelect, select exactly one option.',
  OUTPUT_CONTRACT,
].join('\n');

/**
 * Answer-layer failures; every one of them ends in `next()` (human popup).
 *
 * `NOT_CONFIDENT` and `BAD_LABELS` exist as separate codes because they are the
 * two failures an operator can actually act on: the first says the model judged
 * the excerpt insufficient (working as designed), the second says it produced
 * something that did not line up with the offered options (worth investigating).
 * Collapsing them into one code made a delegate impossible to explain after the
 * fact.
 */
class AnswerError extends Error {
  /**
   * @param {'NO_ADAPTER'|'TIMEOUT'|'ABORTED'|'BAD_OUTPUT'|'BAD_LABELS'|'NOT_CONFIDENT'|'STREAM_ERROR'|'OVERLOAD'} code
   * @param {string} [message]
   * @param {string} [reply] the raw model reply, for the audit trail
   */
  constructor(code, message, reply) {
    super(message ?? `dsh-auto-answer failed: ${code}`);
    this.name = 'AnswerError';
    this.code = code;
    if (reply !== undefined) this.reply = reply;
  }
}

/** How much of a raw model reply the audit trail keeps. */
const REPLY_EXCERPT_CHARS = 600;

/**
 * Normalize an error for logging. `reply` carries the raw model text when the
 * failure was caused by what the model said, which is the only way to explain a
 * delegate after the fact.
 */
function errorDescriptor(err) {
  const code = err && typeof err === 'object' && err.code ? String(err.code) : '';
  const message = err && err.message ? String(err.message) : String(err);
  const reply = err && typeof err === 'object' && typeof err.reply === 'string' ? err.reply : undefined;
  const out = code ? { error: code, message } : { error: 'UNKNOWN', message };
  if (reply !== undefined) out.reply = reply.length > REPLY_EXCERPT_CHARS ? `${reply.slice(0, REPLY_EXCERPT_CHARS)}…` : reply;
  return out;
}

/** Only accept a positive integer. */
function positiveInt(value, fallback) {
  return Number.isInteger(value) && value > 0 ? value : fallback;
}

/** Accept zero as well, for a knob whose "off" position is the number 0. */
function nonNegativeInt(value, fallback) {
  return Number.isInteger(value) && value >= 0 ? value : fallback;
}

/**
 * Failures a second turn can plausibly fix: the model said something that did
 * not satisfy the contract. Everything else is either a transport problem (a
 * retry only burns the shared deadline) or the model's deliberate "the excerpt
 * does not settle this" (NOT_CONFIDENT), which repeating cannot change.
 */
const REPAIRABLE = new Set(['BAD_OUTPUT', 'BAD_LABELS']);

/** How much of a rejected reply a repair turn quotes back. */
const REPAIR_QUOTE_CHARS = 1200;

/**
 * Build the turn that follows a repairable failure: what the model said, why it
 * was rejected, and the output contract once more.
 * @param {AnswerError} err the failure that triggered the repair
 * @returns {string} repair instruction text
 */
function repairPrompt(err) {
  const reason =
    err.code === 'BAD_LABELS'
      ? "it did not answer every question with labels copied from that question's own options"
      : 'it was not the required JSON object';
  const previous =
    typeof err.reply === 'string' && err.reply.trim() !== ''
      ? err.reply.trim().slice(0, REPAIR_QUOTE_CHARS)
      : '(it contained no usable text)';
  return [
    `Your previous reply was rejected because ${reason}.`,
    'What you replied:',
    previous,
    'Answer the same questions again. Copy every label exactly as it appears in the options you were given - never paraphrase and never invent one.',
    OUTPUT_CONTRACT,
  ].join('\n');
}

/**
 * Built-in prompt for the second-opinion judge, used when the primary judge is
 * not confident and `escalation.systemPrompt` is empty. Deliberately short and
 * decision-shaped: it names what outranks what, so the model commits instead of
 * hedging again.
 */
export const DEFAULT_ESCALATION_PROMPT = [
  'You are the second-opinion judge for a DeepSeek Harness session. The first judge returned {"confident": false}; you are called because you can see more of the conversation.',
  'Decide what the OPERATOR of this session would choose, and commit to that choice.',
  'Rules, in order of precedence:',
  "1. The operator's own words outrank every agent report, summary, review or recommendation in the excerpt.",
  '2. Stay consistent with what this project has already written down (glossary, ADRs, coding standards, contribution rules) and with its SDLC - the documented path a change takes from issue to reviewed, gated, released work. Never reverse a settled decision, and never propose skipping a review, a gate or CI to save a step.',
  '3. Never invent facts, paths, credentials, preferences or history the excerpt does not support.',
  '4. Between two options the excerpt supports equally, choose the one that is cheaper to reverse.',
  '5. A label marked "(Recommended)" is a tie-breaker only, never a reason on its own.',
  '6. Answer every question, copy labels exactly as offered, select exactly one option unless the question sets multiSelect, and return confident: true.',
  OUTPUT_CONTRACT,
].join('\n');

/** Appended to the escalation prompt when the human never answered in time. */
const FORCED_SUFFIX = [
  'The operator is unreachable and this question is blocking the session. You must choose now:',
  'return your best option for every question with confident: true, and say in reason which part was a judgement call.',
].join(' ');

/**
 * Normalize the plugin row config. Every field is optional; nothing here throws,
 * so a malformed row degrades into "no provider" and therefore into delegation.
 * @param {object} [raw] plugin row config
 * @returns {Readonly<object>} normalized config
 */
export function normalizeConfig(raw) {
  const c = raw && typeof raw === 'object' ? raw : {};
  const j = c.judge && typeof c.judge === 'object' ? c.judge : {};
  const e = c.escalation && typeof c.escalation === 'object' ? c.escalation : {};
  const f = c.fallback && typeof c.fallback === 'object' ? c.fallback : {};
  const str = (value) => (typeof value === 'string' ? value : '');
  const prompt = (value, fallback) => (typeof value === 'string' && value.trim() !== '' ? value : fallback);
  return Object.freeze({
    enabled: c.enabled !== false,
    judge: Object.freeze({
      provider: str(j.provider),
      model: str(j.model),
      systemPrompt: prompt(j.systemPrompt, DEFAULT_SYSTEM_PROMPT),
      timeoutMs: positiveInt(j.timeoutMs, 60000),
      maxTokens: positiveInt(j.maxTokens, 4096),
      concurrency: positiveInt(j.concurrency, 2),
      contextChars: positiveInt(j.contextChars, 6000),
      repairAttempts: nonNegativeInt(j.repairAttempts, 1),
      maxMessageChars: positiveInt(j.maxMessageChars, 1200),
    }),
    escalation: Object.freeze({
      provider: str(e.provider),
      model: str(e.model),
      systemPrompt: prompt(e.systemPrompt, DEFAULT_ESCALATION_PROMPT),
      timeoutMs: positiveInt(e.timeoutMs, 120000),
      maxTokens: positiveInt(e.maxTokens, 8192),
      contextChars: positiveInt(e.contextChars, 24000),
    }),
    fallback: Object.freeze({
      afterMs: nonNegativeInt(f.afterMs, 600000),
      allowRecommended: f.allowRecommended !== false,
    }),
    auditFile:
      typeof c.auditFile === 'string' && c.auditFile.trim() !== ''
        ? c.auditFile.trim()
        : path.join(os.homedir(), '.dsh', 'logs', 'auto-answer.jsonl'),
  });
}

/** Extract plain text from one message object whose `content` may be blocks. */
function textOfMessage(message) {
  if (message === null || message === undefined) return '';
  if (typeof message === 'string') return message;
  const content = message.content;
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts = [];
  for (const block of content) {
    if (block && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/**
 * Which slice of a user/message event it is, or null to leave it out entirely.
 *
 * The session log delivers a lot more than the operator's own words: relayed
 * subagent reports (5-11 KB each), settle notices, skill catalogs, runtime
 * snapshots. Labelling all of them `[operator]` let one report eat the whole
 * excerpt budget and hid the operator's actual instructions from the judge.
 * @param {object|undefined} source event data.source
 * @returns {'operator'|'agent-report'|'agent-notice'|null} transcript role, or null to skip
 */
function roleOf(source) {
  const kind = source && typeof source.kind === 'string' ? source.kind : '';
  if (kind === '' || kind === 'user') return 'operator';
  if (kind === 'agent-message') return 'agent-report';
  if (kind === 'subagent-settled') return 'agent-notice';
  return null;
}

/**
 * Build a truncated transcript of the session's own message history. Read-only:
 * service messages are skipped, the operator's own words stay distinguishable
 * from relayed agent reports, every message is capped so a few long reports
 * cannot crowd out everything else, and the tail is kept so the most recent
 * context survives the budget.
 * @param {object|undefined} agent the asking agent (waterfall request.agent)
 * @param {number} budget maximum characters of transcript to return
 * @param {number} [maxMessageChars] per-message cap
 * @returns {string} transcript, or '' when unavailable
 */
export function transcriptOf(agent, budget, maxMessageChars = 1200) {
  let events;
  try {
    events = agent && agent.session ? agent.session.events : undefined;
  } catch {
    return '';
  }
  if (!Array.isArray(events)) return '';
  const collected = [];
  for (let i = events.length - 1; i >= 0 && collected.length < 30; i--) {
    const ev = events[i];
    if (!ev || !ev.data) continue;
    let role;
    let message;
    if (ev.type === 'user/message') {
      role = roleOf(ev.data.source);
      if (role === null) continue;
      message = ev.data;
    } else if (ev.type === 'assistant/message') {
      role = 'assistant';
      message = ev.data.message !== undefined ? ev.data.message : ev.data;
    } else {
      continue;
    }
    const text = textOfMessage(message).trim();
    if (text === '') continue;
    const capped =
      text.length > maxMessageChars ? `${text.slice(0, maxMessageChars)}…[+${text.length - maxMessageChars} chars]` : text;
    collected.push(`[${role}] ${capped}`);
  }
  if (collected.length === 0) return '';
  collected.reverse();
  const kept = [];
  let total = 0;
  for (let i = collected.length - 1; i >= 0; i--) {
    total += collected[i].length;
    if (total > budget) break;
    kept.unshift(collected[i]);
  }
  return kept.join('\n\n');
}

/**
 * Extract the first balanced `{...}` object from a model reply and parse it,
 * tolerating ``` fences and surrounding prose.
 * @param {string} text raw model text
 * @returns {object|null} parsed object, or null
 */
export function parseJsonObject(text) {
  if (typeof text !== 'string') return null;
  const cleaned = text.replace(/```/g, '');
  const start = cleaned.indexOf('{');
  if (start === -1) return null;
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < cleaned.length; i++) {
    const ch = cleaned[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) {
        try {
          const value = JSON.parse(cleaned.slice(start, i + 1));
          return value && typeof value === 'object' && !Array.isArray(value) ? value : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

/**
 * Resolve one model-chosen string onto exactly one of a question's option
 * labels.
 *
 * The strictest reading — byte equality — is wrong in practice: asking agents
 * decorate labels with trailing parentheticals such as `(Recommended)` or
 * `(2x usage)`, and a model that drops or reorders that suffix would otherwise
 * be treated as having invented an option. Four widening stages are tried, and
 * **every** stage requires a unique hit: the moment two options could both
 * match, this returns null and the question goes to the human. Tolerance is
 * only ever about formatting, never about guessing which option was meant.
 *
 * @param {string} picked the label the model chose
 * @param {readonly string[]} labels that question's real option labels
 * @returns {string|null} the canonical label, or null when ambiguous/unknown
 */
export function resolveLabel(picked, labels) {
  const normalize = (value) => value.trim().replace(/\s+/g, ' ').toLowerCase();
  const decor = (value) => normalize(value).replace(/\s*\([^)]*\)\s*$/, '').trim();
  const unique = (matches) => (matches.length === 1 ? matches[0] : null);

  const exact = unique(labels.filter((label) => label === picked));
  if (exact !== null) return exact;

  const wanted = normalize(picked);
  const normalized = unique(labels.filter((label) => normalize(label) === wanted));
  if (normalized !== null) return normalized;

  const bare = decor(picked);
  if (bare !== '') {
    const undecorated = unique(labels.filter((label) => decor(label) === bare));
    if (undecorated !== null) return undecorated;
  }

  // Prefix either way, so `Перевантажити метод` still matches
  // `Перевантажити метод і зберегти сумісність (Recommended)` — but only while
  // it stays unambiguous. The length guard keeps a stray one-word answer from
  // matching by accident.
  if (wanted.length >= 4) {
    const prefixed = unique(
      labels.filter((label) => {
        const candidate = normalize(label);
        return candidate.startsWith(wanted) || wanted.startsWith(candidate);
      }),
    );
    if (prefixed !== null) return prefixed;
  }
  return null;
}

/**
 * Turn a model verdict into the seam's answer shape, or null to delegate.
 *
 * Every check is deliberately strict: a verdict that does not answer every
 * question with labels drawn from that question's own options is treated as
 * "cannot answer", never guessed at. Option *formatting* is handled leniently
 * by {@link resolveLabel}; option *identity* never is.
 * @param {object|null} parsed model verdict
 * @param {readonly object[]} questions the questions as asked
 * @returns {{answers: {id: string, selected: string[]}[]}|null}
 */
export function validateVerdict(parsed, questions) {
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
  if (parsed.confident !== true) return null;
  if (!Array.isArray(parsed.answers)) return null;
  const byId = new Map();
  for (const entry of parsed.answers) {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return null;
    if (typeof entry.id !== 'string' || byId.has(entry.id)) return null;
    if (!Array.isArray(entry.selected) || entry.selected.some((s) => typeof s !== 'string')) return null;
    byId.set(entry.id, entry.selected);
  }
  const answers = [];
  for (const question of questions) {
    const selected = byId.get(question.id);
    if (selected === undefined || selected.length === 0) return null;
    const labels = (question.options ?? [])
      .map((option) => (option && typeof option.label === 'string' ? option.label : null))
      .filter((label) => label !== null);
    const canonical = [];
    for (const want of selected) {
      const hit = resolveLabel(want, labels);
      if (hit === null) return null;
      if (!canonical.includes(hit)) canonical.push(hit);
    }
    if (question.multiSelect !== true && canonical.length !== 1) return null;
    answers.push({ id: question.id, selected: canonical });
  }
  return { answers };
}

/** Abort helper that keeps upstream cancellation distinct from our timeout. */
function throwIfAborted(streamSignal, upstream) {
  if (streamSignal.aborted || (upstream && upstream.aborted)) throw new AnswerError('ABORTED');
}

/**
 * Run one model call and turn its reply into the seam's answer shape.
 *
 * Split out of {@link createAnswerer} so a repair turn re-runs exactly this
 * against the same conversation with one more message appended.
 *
 * @param {object} options
 * @param {object} options.llm the `ctx.llm` service
 * @param {object[]} options.messages conversation so far
 * @param {string} options.provider provider route
 * @param {string} options.model model id
 * @param {string} options.systemPrompt system prompt
 * @param {number} options.maxTokens output budget
 * @param {AbortSignal} options.streamSignal deadline-scoped signal
 * @param {AbortSignal} [options.upstream] the caller's own signal
 * @param {readonly object[]} options.questions the questions as asked
 * @returns {Promise<{answers: object[]}>} validated verdict
 */
async function attemptAnswer({ llm, messages, provider, model, systemPrompt, maxTokens, streamSignal, upstream, questions }) {
  const assembler = new BlockAssembler();
  try {
    for await (const chunk of llm.stream({ provider, model, messages, system: systemPrompt, maxTokens, signal: streamSignal })) {
      throwIfAborted(streamSignal, upstream);
      assembler.push(chunk);
    }
    throwIfAborted(streamSignal, upstream);
  } catch (err) {
    throwIfAborted(streamSignal, upstream);
    if (err instanceof AnswerError) throw err;
    throw new AnswerError('STREAM_ERROR', `dsh-auto-answer stream threw: ${err && err.message ? err.message : String(err)}`);
  }
  const blocks = assembler.blocks();
  const text = blocks
    .filter((b) => b.type === 'text')
    .map((b) => b.text)
    .join('\n');
  // Text is read before the tool-call check so a rejected reply carries whatever
  // the model did say alongside the call - the audit line and the repair turn
  // both quote it.
  if (blocks.some((b) => b.type === 'tool-call')) {
    throw new AnswerError('BAD_OUTPUT', 'reply contained a tool-call block', text.trim() === '' ? undefined : text);
  }
  if (typeof text !== 'string' || text.trim() === '') throw new AnswerError('BAD_OUTPUT', 'reply contained no text');
  const parsed = parseJsonObject(text);
  if (parsed === null) throw new AnswerError('BAD_OUTPUT', 'reply was not a JSON object', text);
  if (parsed.confident !== true) {
    throw new AnswerError('NOT_CONFIDENT', 'the model judged the excerpt insufficient to answer', text);
  }
  const verdict = validateVerdict(parsed, questions);
  if (verdict === null) {
    throw new AnswerError('BAD_LABELS', 'reply did not answer every question with valid option labels', text);
  }
  return verdict;
}

/**
 * Run one judge profile: its own deadline, its own repair turns, no shared state.
 *
 * Profiles are independent on purpose. The second-opinion call happens after the
 * primary one already spent its deadline, so reusing that deadline would abort
 * the escalation before it started.
 *
 * @param {object} options
 * @param {object} options.llm the `ctx.llm` service
 * @param {object} options.profile `{provider, model, systemPrompt, timeoutMs, maxTokens, repairAttempts}`
 * @param {object} options.input payload for the model
 * @param {AbortSignal} [options.signal] the request's own signal
 * @param {object[]} [options.trace] collects one descriptor per repair turn
 * @param {boolean} [options.forced] append the "operator unreachable, choose now" suffix
 * @returns {Promise<{answers: object[]}>} validated verdict
 */
async function runProfile({ llm, profile, input, signal, trace, forced }) {
  const handle = deadline(signal, profile.timeoutMs, TIMEOUT_CODE);
  try {
    const streamSignal = handle.signal;
    throwIfAborted(streamSignal, signal);
    const systemPrompt = forced === true ? `${profile.systemPrompt}\n\n${FORCED_SUFFIX}` : profile.systemPrompt;
    const messages = [
      createUserMessage({
        content: [{ type: 'text', text: JSON.stringify(input, null, 2) }],
        source: { kind: 'plugin', plugin: 'dsh-auto-answer' },
      }),
    ];
    const attempt = {
      llm,
      messages,
      provider: profile.provider,
      model: profile.model,
      systemPrompt,
      maxTokens: profile.maxTokens,
      streamSignal,
      upstream: signal,
      questions: input.questions,
    };
    for (let repairs = 0; ; repairs++) {
      try {
        return await attemptAnswer(attempt);
      } catch (err) {
        const fixable = err instanceof AnswerError && REPAIRABLE.has(err.code);
        if (!fixable || repairs >= profile.repairAttempts) throw err;
        if (Array.isArray(trace)) trace.push({ attempt: repairs + 1, ...errorDescriptor(err) });
        messages.push(
          createUserMessage({
            content: [{ type: 'text', text: repairPrompt(err) }],
            source: { kind: 'plugin', plugin: 'dsh-auto-answer' },
          }),
        );
      }
    }
  } finally {
    handle[Symbol.dispose]();
  }
}

/**
 * Last-resort answer that needs no model at all: the single option marked
 * `(Recommended)`, or the only option there is. Returns null when a question
 * offers a real choice, because then the pick would be a coin flip.
 * @param {readonly object[]} questions the questions as asked
 * @returns {{answers: object[]}|null} a complete answer, or null
 */
export function pickObvious(questions) {
  const answers = [];
  for (const question of questions) {
    const labels = (question.options ?? [])
      .map((option) => (option && typeof option.label === 'string' ? option.label : null))
      .filter((label) => label !== null);
    const recommended = labels.filter((label) => /\(\s*(recommended|рекомендован\w*|рекомендовано)\s*\)/i.test(label));
    const pick = recommended.length === 1 ? recommended[0] : labels.length === 1 ? labels[0] : null;
    if (pick === null) return null;
    answers.push({ id: question.id, selected: [pick] });
  }
  return answers.length > 0 ? { answers } : null;
}

/**
 * Create the two judge entry points.
 *
 * `answer` runs the primary judge and, when it comes back not confident, the
 * second-opinion judge with more of the conversation. `force` is the same
 * second-opinion call with the "the operator is unreachable, choose now" suffix,
 * for the bounded-wait fallback.
 *
 * @param {object} options
 * @param {object} options.llm the `ctx.llm` service
 * @param {object} options.config normalized plugin config
 * @returns {{answer: Function, force: Function}} both return `{verdict, via}`
 */
export function createAnswerer({ llm, config }) {
  const judgeProfile = { ...config.judge };
  const escalationProfile = { ...config.escalation, repairAttempts: config.judge.repairAttempts };
  const hasEscalation = config.escalation.provider !== '' && config.escalation.model !== '';
  const limit = config.judge.concurrency;
  let active = 0;

  async function call(profile, input, signal, trace, forced) {
    if (!llm || typeof llm.stream !== 'function') throw new AnswerError('NO_ADAPTER', 'llm service missing or lacks stream()');
    if (active >= limit) throw new AnswerError('OVERLOAD', `dsh-auto-answer at concurrency limit ${limit}`);
    active++;
    try {
      return await runProfile({ llm, profile, input, signal, trace, forced });
    } finally {
      active--;
    }
  }

  return {
    /**
     * @param {{primary: object, escalation?: object}} payloads the payload per profile
     * @returns {Promise<{verdict: object, via: 'judge'|'escalation'}>}
     */
    async answer(payloads, signal, trace) {
      try {
        return { verdict: await call(judgeProfile, payloads.primary, signal, trace, false), via: 'judge' };
      } catch (err) {
        const unsure = err instanceof AnswerError && err.code === 'NOT_CONFIDENT';
        if (!hasEscalation || !unsure) throw err;
        // A second opinion is exactly what this verdict asks for: the primary
        // judge said the excerpt does not settle it, and this profile sees more.
        if (Array.isArray(trace)) trace.push({ attempt: 'escalation', ...errorDescriptor(err) });
        const verdict = await call(escalationProfile, payloads.escalation ?? payloads.primary, signal, trace, false);
        return { verdict, via: 'escalation' };
      }
    },
    /**
     * @param {{primary: object, escalation?: object}} payloads the payload per profile
     * @returns {Promise<{verdict: object, via: 'fallback-judge'|'fallback-escalation'}>}
     */
    async force(payloads, signal, trace) {
      const profile = hasEscalation ? escalationProfile : judgeProfile;
      const input = hasEscalation ? payloads.escalation ?? payloads.primary : payloads.primary;
      const verdict = await call(profile, input, signal, trace, true);
      return { verdict, via: hasEscalation ? 'fallback-escalation' : 'fallback-judge' };
    },
  };
}

/**
 * Plugin body.
 * @param {import('@deepseek-ai/cordis').Context} ctx
 * @param {object} [rawConfig] plugin row config
 */
export function apply(ctx, rawConfig) {
  const config = normalizeConfig(rawConfig);
  const logger = ctx.logger('auto-answer');
  const judge = createAnswerer({ llm: ctx.llm, config });

  let dirEnsured = false;
  function audit(entry) {
    if (!dirEnsured) {
      dirEnsured = true;
      try {
        fs.mkdirSync(path.dirname(config.auditFile), { recursive: true });
      } catch (err) {
        logger.warn('cannot create audit directory', errorDescriptor(err));
      }
    }
    try {
      fs.promises.appendFile(config.auditFile, JSON.stringify(entry) + '\n', 'utf8').catch((err) => {
        logger.warn('audit append failed', errorDescriptor(err));
      });
    } catch (err) {
      logger.warn('audit append failed', errorDescriptor(err));
    }
  }

  if (!config.enabled) {
    logger.info('disabled by config; every question goes to the human UI');
    return;
  }
  if (config.judge.provider === '' || config.judge.model === '') {
    logger.warn('no judge provider/model configured; every question goes to the human UI');
  }

  /**
   * Waterfall listener. Runs before the browser answerer. It either claims the
   * request with a judge verdict, or hands it to the human popup — but never
   * waits for that human for ever: `fallback.afterMs` bounds the wait, and the
   * timeout answers from the second-opinion profile.
   */
  const handler = async (request, next) => {
    const questions = request && Array.isArray(request.questions) ? request.questions : [];
    if (questions.length === 0) return next();
    // Option-only gate: a free-text question needs facts only the operator has.
    if (!questions.every((q) => q && Array.isArray(q.options) && q.options.length > 0)) return next();
    if (config.judge.provider === '' || config.judge.model === '') return next();

    const agent = request.agent;
    const asked = questions.map((q) => ({
      id: q.id,
      question: q.question,
      ...(q.header !== undefined ? { header: q.header } : {}),
      options: (q.options ?? []).map((o) => ({
        label: o && o.label,
        ...(o && o.description !== undefined ? { description: o.description } : {}),
      })),
      ...(q.multiSelect !== undefined ? { multiSelect: q.multiSelect } : {}),
    }));
    // The two profiles differ in how much of the conversation they see: the
    // second opinion is called precisely because the first one lacked context.
    const payloadFor = (contextChars) => {
      const excerpt = transcriptOf(agent, contextChars, config.judge.maxMessageChars);
      return { questions: asked, ...(excerpt !== '' ? { conversationExcerpt: excerpt } : {}) };
    };
    const payloads = { primary: payloadFor(config.judge.contextChars) };
    if (config.escalation.provider !== '' && config.escalation.model !== '') {
      payloads.escalation = payloadFor(config.escalation.contextChars);
    }

    const base = {
      time: Date.now(),
      sessionId: agent && agent.session ? agent.session.id : undefined,
      questionIds: questions.map((q) => q.id),
    };

    // Repair turns and escalations are recorded so an "answered" line can still
    // show what it took, and a "delegate" line how far it got.
    const trace = [];
    const claim = (verdict, via, extra) => {
      logger.info('answered ask_user_question', { questionIds: base.questionIds, via, answers: verdict.answers, trace: trace.length });
      audit({
        ...base,
        outcome: 'answered',
        via,
        answers: verdict.answers,
        ...(trace.length > 0 ? { trace } : {}),
        ...extra,
      });
      return verdict;
    };
    const delegate = (descriptor, extra) => {
      logger.warn('delegating ask_user_question to the human UI', { ...descriptor, trace: trace.length, ...extra });
      audit({ ...base, outcome: 'delegate', ...(trace.length > 0 ? { trace } : {}), ...extra, ...descriptor });
    };

    let failure;
    try {
      const { verdict, via } = await judge.answer(payloads, request.signal, trace);
      return claim(verdict, via);
    } catch (err) {
      // The descriptor carries `reply` when the model's own text caused the
      // failure — without it a delegate cannot be explained after the fact.
      failure = errorDescriptor(err);
    }

    // The models are done; the question is the operator's now — but not for
    // ever. The popup races a timer so a sleeping operator cannot stall a
    // session the way an unbounded wait used to.
    const human = next();
    if (config.fallback.afterMs <= 0) {
      delegate(failure);
      return human;
    }

    const TIMED_OUT = Symbol('fallback-timeout');
    let timer;
    let winner;
    try {
      winner = await Promise.race([
        human,
        new Promise((resolve) => {
          timer = setTimeout(() => resolve(TIMED_OUT), config.fallback.afterMs);
          if (typeof timer.unref === 'function') timer.unref();
        }),
      ]);
    } catch (err) {
      // The popup path failed on its own terms (aborted turn, closed UI);
      // nothing to add, and the caller must see the original rejection.
      logger.warn('human answerer rejected', errorDescriptor(err));
      throw err;
    } finally {
      clearTimeout(timer);
    }
    if (winner !== TIMED_OUT) {
      // The operator answered inside the window: the delegate is written here,
      // once, so the trail keeps exactly one line per request.
      delegate(failure);
      return winner;
    }

    human.then(
      () => logger.info('operator answered after the fallback had already claimed the question', { questionIds: base.questionIds }),
      () => {},
    );
    const waited = { afterMs: config.fallback.afterMs };
    try {
      const { verdict, via } = await judge.force(payloads, request.signal, trace);
      return claim(verdict, via, { fallback: waited });
    } catch (err) {
      const forcedError = errorDescriptor(err);
      if (config.fallback.allowRecommended) {
        const obvious = pickObvious(questions);
        if (obvious !== null) return claim(obvious, 'fallback-recommended', { fallback: { ...waited, forcedError } });
      }
      // Nothing safe to pick and nobody to ask: the question stays with the
      // operator, and the audit says why this wait is unbounded again.
      logger.warn('fallback could not answer; the question keeps waiting for the operator', forcedError);
      delegate(failure, { fallback: 'unanswered', forcedError });
      return human;
    }
  };

  // All side effects are owned by ctx.effect, so unloading removes the listener.
  ctx.effect(
    () => ctx.on('user-questions/request', handler, { prepend: true }),
    'auto-answer: user-questions answerer',
  );
}
