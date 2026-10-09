/**
 * Real-Cordis probe for dsh-auto-answer.
 *
 * Boots a genuine cordis Context with a fake `llm` service, mounts the plugin,
 * then drives the `user-questions/request` waterfall the same way
 * `ctx.userQuestions.ask()` does. Asserts both halves of the contract:
 *   - a confident, well-formed verdict CLAIMS the request (the human popup
 *     answerer never runs);
 *   - every doubtful path DELEGATES (the human popup answerer runs).
 *
 * Run: node --test test/probe.test.mjs
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { Context, Service } from '@deepseek-ai/cordis';

import { name, inject, apply } from '../lib/index.js';

/** Fake `llm` service: `stream()` delegates to the test's producer. */
class FakeLlm extends Service {
  constructor(ctx) {
    super(ctx, 'llm');
    this.produce = undefined;
    this.calls = [];
  }

  stream(options) {
    this.calls.push(options);
    const produce = this.produce;
    if (typeof produce !== 'function') return (async function* empty() {})();
    return produce(options);
  }
}

/** Encode one text reply as the StreamChunk protocol BlockAssembler consumes. */
function textChunks(text) {
  return [
    { type: 'block-start', index: 0, blockType: 'text' },
    { type: 'text-delta', index: 0, text },
    { type: 'block-end', index: 0, block: { type: 'text', text } },
    { type: 'usage', usage: { inputTokens: 2, outputTokens: 3 } },
    { type: 'finish', reason: { kind: 'stop' } },
  ];
}

/** Stream chunks from an array of replies (need not be used). */
async function* fromChunks(chunks) {
  for (const chunk of chunks) yield chunk;
}

/**
 * Fake producer that walks a script of replies, one per model call, holding the
 * last one once the script runs out.
 * @param {string[]} replies replies in call order
 * @returns {Function} FakeLlm producer
 */
function scripted(replies) {
  let call = 0;
  return () => fromChunks(textChunks(replies[Math.min(call++, replies.length - 1)]));
}

/**
 * Read the audit trail once it holds at least `atLeast` complete lines.
 *
 * The plugin appends fire-and-forget, so a fixed sleep can read a half-written
 * line; a truncated line fails JSON.parse and the poll simply retries.
 */
async function readAuditLines(file, atLeast = 1, timeoutMs = 3000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try {
      const text = fs.readFileSync(file, 'utf8').trim();
      if (text !== '') {
        const lines = text.split('\n').map((line) => JSON.parse(line));
        if (lines.length >= atLeast) return lines;
      }
    } catch {
      /* partial append: retry */
    }
    if (Date.now() > deadline) throw new Error(`audit file did not reach ${atLeast} complete line(s) within ${timeoutMs}ms`);
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

let auditSeq = 0;
function auditPath() {
  auditSeq += 1;
  return path.join(os.tmpdir(), `auto-answer-probe-${process.pid}-${auditSeq}.jsonl`);
}

/**
 * Boot a real cordis Context with the plugin mounted.
 * @param {object} config plugin row config
 * @param {Function} [produce] FakeLlm stream producer
 * @returns {Promise<{ctx: Context, llm: FakeLlm, uiCalls: object[]}>} booted context
 */
async function boot(config, produce) {
  const ctx = new Context();
  await ctx.plugin(FakeLlm, {});
  const llm = ctx.get('llm');
  llm.produce = produce;
  // The browser answerer, registered AFTER the plugin: it must run only when the
  // plugin delegates. (The plugin registers with { prepend: true }.)
  const uiCalls = [];
  ctx.on('user-questions/request', (request, next) => {
    uiCalls.push(request);
    return next();
  });
  await ctx.plugin({ name, inject, apply }, config);
  return { ctx, llm, uiCalls };
}

const QUESTIONS = [
  {
    id: 'preset',
    question: 'Which preset do you want?',
    header: 'Preset',
    options: [{ label: 'balanced (Recommended)' }, { label: 'permissive' }, { label: 'yolo' }],
  },
];

function request(questions, agent) {
  return { questions, ...(agent !== undefined ? { agent } : {}) };
}

/** Fallback the waterfall uses when every listener delegates. */
function humanFallback() {
  return { answers: [{ id: 'human', selected: ['answered-by-human'] }] };
}

function baseConfig(extra = {}) {
  return {
    enabled: true,
    judge: { provider: 'example-provider', model: 'example-model', timeoutMs: 5000, maxTokens: 512, concurrency: 2 },
    auditFile: auditPath(),
    ...extra,
  };
}

/* ------------------------------------------------------------------ *
 * claiming the request
 * ------------------------------------------------------------------ */

test('probe: a confident verdict claims the request and the UI never runs', async () => {
  const { ctx, llm, uiCalls } = await boot(baseConfig(), () =>
    fromChunks(textChunks('```json\n{"answers":[{"id":"preset","selected":["permissive"]}],"confident":true,"reason":"оператор обрав його раніше"}\n```')),
  );
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'preset', selected: ['permissive'] }] });
  assert.equal(uiCalls.length, 0, 'human popup must not be reached');
  assert.equal(llm.calls.length, 1);
  assert.equal(llm.calls[0].provider, 'example-provider');
  assert.equal(llm.calls[0].model, 'example-model');
  assert.equal(llm.calls[0].maxTokens, 512);
});

test('probe: the prompt carries the questions and the conversation excerpt', async () => {
  const { ctx, llm } = await boot(baseConfig(), () =>
    fromChunks(textChunks('{"answers":[{"id":"preset","selected":["yolo"]}],"confident":true,"reason":"ок"}')),
  );
  const agent = {
    session: {
      id: 'session-probe',
      events: [
        { type: 'user/message', data: { content: [{ type: 'text', text: 'хочу менше попапів' }] } },
        { type: 'assistant/message', data: { message: { content: [{ type: 'text', text: 'зрозумів' }] } } },
      ],
    },
  };
  await ctx.waterfall('user-questions/request', request(QUESTIONS, agent), humanFallback);
  const sent = llm.calls[0].messages[0].content[0].text;
  const payload = JSON.parse(sent);
  assert.equal(payload.questions[0].id, 'preset');
  assert.equal(payload.questions[0].options.length, 3);
  assert.match(payload.conversationExcerpt, /хочу менше попапів/);
  assert.match(payload.conversationExcerpt, /\[assistant\] зрозумів/);
});

/* ------------------------------------------------------------------ *
 * delegating to the human
 * ------------------------------------------------------------------ */

const DELEGATING = [
  ['free-text question (no options)', request([{ id: 'k', question: 'Which path?' }]), undefined],
  ['mixed set: one question lacks options', request([QUESTIONS[0], { id: 'k', question: 'Which path?' }]), undefined],
  ['no questions at all', request([]), undefined],
];

for (const [label, req, produce] of DELEGATING) {
  test(`probe: delegates — ${label}`, async () => {
    const { ctx, llm, uiCalls } = await boot(baseConfig(), produce);
    const result = await ctx.waterfall('user-questions/request', req, humanFallback);
    assert.deepEqual(result, { answers: [{ id: 'human', selected: ['answered-by-human'] }] });
    assert.equal(uiCalls.length, 1, 'human popup must answer');
    assert.equal(llm.calls.length, 0, 'no model call for an ineligible request');
  });
}

const DOUBTFUL = [
  ['model is not confident', '{"answers":[{"id":"preset","selected":["yolo"]}],"confident":false}'],
  ['model invents an option label', '{"answers":[{"id":"preset","selected":["turbo"]}],"confident":true}'],
  ['model skips a question', '{"answers":[],"confident":true}'],
  ['model picks two for a single-select', '{"answers":[{"id":"preset","selected":["yolo","permissive"]}],"confident":true}'],
  ['reply is not JSON', 'I think permissive is best.'],
  ['reply is empty', ''],
];

for (const [label, reply] of DOUBTFUL) {
  test(`probe: delegates — ${label}`, async () => {
    const { ctx, uiCalls } = await boot(baseConfig(), () => fromChunks(textChunks(reply)));
    const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
    assert.deepEqual(result, { answers: [{ id: 'human', selected: ['answered-by-human'] }] });
    assert.equal(uiCalls.length, 1);
  });
}

test('probe: delegates when the model stream throws', async () => {
  const { ctx, uiCalls } = await boot(baseConfig(), () =>
    (async function* boom() {
      yield { type: 'block-start', index: 0, blockType: 'text' };
      throw new Error('upstream exploded');
    })(),
  );
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'human', selected: ['answered-by-human'] }] });
  assert.equal(uiCalls.length, 1);
});

test('probe: delegates when no provider/model is configured', async () => {
  const { ctx, llm, uiCalls } = await boot(
    { ...baseConfig(), judge: { provider: '', model: '' } },
    () => fromChunks(textChunks('{"answers":[{"id":"preset","selected":["yolo"]}],"confident":true}')),
  );
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'human', selected: ['answered-by-human'] }] });
  assert.equal(uiCalls.length, 1);
  assert.equal(llm.calls.length, 0);
});

test('probe: delegates when disabled by config', async () => {
  const { ctx, llm, uiCalls } = await boot(
    { ...baseConfig(), enabled: false },
    () => fromChunks(textChunks('{"answers":[{"id":"preset","selected":["yolo"]}],"confident":true}')),
  );
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'human', selected: ['answered-by-human'] }] });
  assert.equal(uiCalls.length, 1);
  assert.equal(llm.calls.length, 0);
});

/* ------------------------------------------------------------------ *
 * repair pass
 * ------------------------------------------------------------------ */

const GOOD = '{"answers":[{"id":"preset","selected":["permissive"]}],"confident":true,"reason":"ок"}';

test('probe: a malformed first reply is repaired and the request is still claimed', async () => {
  const file = auditPath();
  const { ctx, llm, uiCalls } = await boot(
    { ...baseConfig(), auditFile: file },
    scripted(['I think permissive is best.', GOOD]),
  );
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'preset', selected: ['permissive'] }] });
  assert.equal(uiCalls.length, 0, 'human popup must not be reached');
  assert.equal(llm.calls.length, 2, 'exactly one repair turn');
  const lines = await readAuditLines(file, 1);
  assert.equal(lines[0].outcome, 'answered');
  assert.equal(lines[0].trace.length, 1, 'the answered line still shows the rejected reply');
  assert.equal(lines[0].trace[0].error, 'BAD_OUTPUT');
  fs.rmSync(file, { force: true });
});

test('probe: the repair turn quotes the rejected reply and restates the contract', async () => {
  // The conversation array is reused across attempts, so record its length from
  // inside the producer: after the repair it is no longer 1 for the first call.
  const lengths = [];
  const { ctx, llm } = await boot(baseConfig(), (options) => {
    lengths.push(options.messages.length);
    return fromChunks(textChunks(lengths.length === 1 ? 'two' : GOOD));
  });
  await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(lengths, [1, 2], 'the first call carries the questions alone; the repair extends that conversation');
  const repair = llm.calls[1].messages[1].content[0].text;
  assert.match(repair, /previous reply was rejected/i);
  assert.match(repair, /two/, 'the rejected reply is quoted back');
  assert.match(repair, /"answers"/, 'the output contract is restated');
});

test('probe: a reply with invented labels is repaired', async () => {
  const { ctx, llm, uiCalls } = await boot(
    baseConfig(),
    scripted(['{"answers":[{"id":"preset","selected":["turbo"]}],"confident":true}', '{"answers":[{"id":"preset","selected":["balanced"]}],"confident":true,"reason":"ок"}']),
  );
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'preset', selected: ['balanced (Recommended)'] }] }, 'canonical label returned');
  assert.equal(uiCalls.length, 0, 'human popup must not be reached');
  assert.equal(llm.calls.length, 2);
});

test('probe: gives up after the configured repair attempt', async () => {
  const file = auditPath();
  const { ctx, llm, uiCalls } = await boot({ ...baseConfig(), auditFile: file }, () => fromChunks(textChunks('still not json')));
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'human', selected: ['answered-by-human'] }] });
  assert.equal(uiCalls.length, 1, 'the popup answers once the repairs are spent');
  assert.equal(llm.calls.length, 2, 'one attempt plus one repair, never more');
  const lines = await readAuditLines(file, 1);
  assert.equal(lines[0].outcome, 'delegate');
  assert.equal(lines[0].error, 'BAD_OUTPUT');
  assert.equal(lines[0].trace.length, 1);
  fs.rmSync(file, { force: true });
});

test('probe: repairAttempts buys exactly that many extra turns', async () => {
  const { ctx, llm, uiCalls } = await boot(
    { ...baseConfig(), judge: { ...baseConfig().judge, repairAttempts: 3 } },
    scripted(['nope', 'still nope', 'not json either', GOOD]),
  );
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'preset', selected: ['permissive'] }] });
  assert.equal(uiCalls.length, 0);
  assert.equal(llm.calls.length, 4, 'three repairs after the first attempt');
});

test('probe: repairAttempts: 0 disables the repair pass', async () => {
  const { ctx, llm, uiCalls } = await boot(
    { ...baseConfig(), judge: { ...baseConfig().judge, repairAttempts: 0 } },
    scripted(['nope', GOOD]),
  );
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'human', selected: ['answered-by-human'] }] });
  assert.equal(uiCalls.length, 1);
  assert.equal(llm.calls.length, 1, 'the scripted second reply is never asked for');
});

test('probe: an unsure verdict is never repaired', async () => {
  const { ctx, llm, uiCalls } = await boot(baseConfig(), scripted(['{"confident":false}', GOOD]));
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'human', selected: ['answered-by-human'] }] });
  assert.equal(uiCalls.length, 1);
  assert.equal(llm.calls.length, 1, 'repeating the question cannot change "not confident"');
});

/* ------------------------------------------------------------------ *
 * second opinion (escalation)
 * ------------------------------------------------------------------ */

/** Never-answering operator: the popup is open, the human is asleep. */
function asleepOperator() {
  return new Promise(() => {});
}

function escalationConfig(extra = {}) {
  return {
    ...baseConfig(),
    escalation: { provider: 'big-provider', model: 'big-model', timeoutMs: 5000, maxTokens: 512, contextChars: 24000 },
    ...extra,
  };
}

test('probe: a second opinion claims a question the first judge was unsure about', async () => {
  const file = auditPath();
  const { ctx, llm, uiCalls } = await boot(
    escalationConfig({ auditFile: file }),
    scripted(['{"confident":false}', '{"answers":[{"id":"preset","selected":["balanced"]}],"confident":true,"reason":"ок"}']),
  );
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'preset', selected: ['balanced (Recommended)'] }] });
  assert.equal(uiCalls.length, 0, 'the operator is never bothered');
  assert.equal(llm.calls.length, 2);
  assert.equal(llm.calls[1].provider, 'big-provider', 'the second call goes to the escalation profile');
  assert.equal(llm.calls[1].model, 'big-model');
  const lines = await readAuditLines(file, 1);
  assert.equal(lines[0].outcome, 'answered');
  assert.equal(lines[0].via, 'escalation');
  assert.equal(lines[0].trace.length, 1, 'the unsure verdict is on record');
  fs.rmSync(file, { force: true });
});

test('probe: the second opinion sees more of the conversation', async () => {
  // Twenty short messages: the primary budget (6000) fits about six, the
  // escalation budget (24000) fits all of them.
  const agent = {
    session: {
      id: 'session-big',
      events: Array.from({ length: 20 }, (_, i) => ({
        type: 'user/message',
        data: { content: `m${i}-${'x'.repeat(1000)}` },
      })),
    },
  };
  const { ctx, llm } = await boot(escalationConfig(), scripted(['{"confident":false}', '{"answers":[{"id":"preset","selected":["yolo"]}],"confident":true,"reason":"ок"}']));
  await ctx.waterfall('user-questions/request', request(QUESTIONS, agent), humanFallback);
  const first = JSON.parse(llm.calls[0].messages[0].content[0].text).conversationExcerpt;
  const second = JSON.parse(llm.calls[1].messages[0].content[0].text).conversationExcerpt;
  assert.ok(second.length > first.length, `escalation excerpt (${second.length}) must exceed the primary one (${first.length})`);
});

test('probe: a still-unsure second opinion falls through to the operator', async () => {
  const { ctx, llm, uiCalls } = await boot(
    escalationConfig(),
    scripted(['{"confident":false}', '{"confident":false}']),
  );
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'human', selected: ['answered-by-human'] }] });
  assert.equal(uiCalls.length, 1);
  assert.equal(llm.calls.length, 2);
});

/* ------------------------------------------------------------------ *
 * bounded wait (fallback)
 * ------------------------------------------------------------------ */

test('probe: the wait for the operator is bounded and the fallback answers', async () => {
  const file = auditPath();
  const { ctx, llm, uiCalls } = await boot(
    escalationConfig({ auditFile: file, fallback: { afterMs: 60 } }),
    scripted([
      '{"confident":false}',
      '{"confident":false}',
      '{"answers":[{"id":"preset","selected":["yolo"]}],"confident":true,"reason":"вимушений вибір"}',
    ]),
  );
  const started = Date.now();
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), asleepOperator);
  assert.deepEqual(result, { answers: [{ id: 'preset', selected: ['yolo'] }] }, 'the fallback answered');
  assert.ok(Date.now() - started < 5000, 'the call returned once the window closed');
  assert.equal(uiCalls.length, 1, 'the popup was still opened for the operator');
  assert.equal(llm.calls.length, 3, 'unsure, unsure again, then the forced pick');
  assert.match(llm.calls[2].system, /unreachable/i, 'the forced call carries the forced suffix');
  const lines = await readAuditLines(file, 1);
  assert.equal(lines[0].outcome, 'answered');
  assert.equal(lines[0].via, 'fallback-escalation');
  assert.equal(lines[0].fallback.afterMs, 60);
  fs.rmSync(file, { force: true });
});

test('probe: when no model can force a pick, the (Recommended) option is taken', async () => {
  const file = auditPath();
  const { ctx } = await boot({ ...baseConfig(), auditFile: file, fallback: { afterMs: 50 } }, () => fromChunks(textChunks('not json')));
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), asleepOperator);
  assert.deepEqual(result, { answers: [{ id: 'preset', selected: ['balanced (Recommended)'] }] });
  const lines = await readAuditLines(file, 1);
  assert.equal(lines[0].via, 'fallback-recommended');
  assert.equal(lines[0].fallback.afterMs, 50);
  fs.rmSync(file, { force: true });
});

test('probe: a prompt operator answer is never overridden', async () => {
  const file = auditPath();
  const { ctx, llm } = await boot(
    escalationConfig({ auditFile: file, fallback: { afterMs: 30000 } }),
    () => fromChunks(textChunks('not json')),
  );
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'human', selected: ['answered-by-human'] }] });
  assert.equal(llm.calls.length, 2, 'no forced call once the operator answered');
  const lines = await readAuditLines(file, 1);
  assert.equal(lines[0].outcome, 'delegate');
  assert.equal(lines[0].via, undefined, 'nothing claimed it');
  fs.rmSync(file, { force: true });
});

test('probe: fallback.afterMs: 0 keeps the unbounded wait', async () => {
  const { ctx, llm } = await boot(
    escalationConfig({ fallback: { afterMs: 0 } }),
    () => fromChunks(textChunks('not json')),
  );
  const late = async () => {
    await new Promise((resolve) => setTimeout(resolve, 120));
    return humanFallback();
  };
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), late);
  assert.deepEqual(result, { answers: [{ id: 'human', selected: ['answered-by-human'] }] }, 'the late human answer still wins');
  assert.equal(llm.calls.length, 2, 'the two judge calls and nothing else');
});

/* ------------------------------------------------------------------ *
 * audit trail
 * ------------------------------------------------------------------ */

test('probe: writes an audit line for both outcomes', async () => {
  const file = auditPath();
  const confident = await boot({ ...baseConfig(), auditFile: file }, () =>
    fromChunks(textChunks('{"answers":[{"id":"preset","selected":["yolo"]}],"confident":true,"reason":"ок"}')),
  );
  await confident.ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  // Appends are fire-and-forget, so settle the first line before the second run
  // starts writing: otherwise the two appends can land out of order.
  await readAuditLines(file, 1);

  const doubtful = await boot({ ...baseConfig(), auditFile: file }, () => fromChunks(textChunks('nope')));
  await doubtful.ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);

  const lines = await readAuditLines(file, 2);
  assert.deepEqual(lines.map((l) => l.outcome), ['answered', 'delegate']);
  assert.equal(lines[1].error, 'BAD_OUTPUT');
  assert.deepEqual(lines[0].answers, [{ id: 'preset', selected: ['yolo'] }]);
  fs.rmSync(file, { force: true });
});

test('probe: dropping the "(Recommended)" decoration from a pick still claims the request', async () => {
  // QUESTIONS offers 'balanced (Recommended)'; a model that echoes only
  // 'balanced' used to be treated as inventing an option and losing the
  // request to a popup.
  const { ctx, uiCalls } = await boot(baseConfig(), () =>
    fromChunks(textChunks('{"answers":[{"id":"preset","selected":["balanced"]}],"confident":true,"reason":"ок"}')),
  );
  const result = await ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  assert.deepEqual(result, { answers: [{ id: 'preset', selected: ['balanced (Recommended)'] }] }, 'canonical label returned');
  assert.equal(uiCalls.length, 0, 'human popup must not be reached');
});

test('probe: the audit names the failure and keeps the raw reply', async () => {
  const file = auditPath();
  const unsure = await boot({ ...baseConfig(), auditFile: file }, () => fromChunks(textChunks('{"confident":false}')));
  await unsure.ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);
  // See the note above: settle each append before the next one is issued.
  await readAuditLines(file, 1);

  const badLabel = await boot({ ...baseConfig(), auditFile: file }, () =>
    fromChunks(textChunks('{"answers":[{"id":"preset","selected":["turbo"]}],"confident":true}')),
  );
  await badLabel.ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);

  const lines = await readAuditLines(file, 2);
  assert.deepEqual(lines.map((l) => l.error), ['NOT_CONFIDENT', 'BAD_LABELS'], 'the two delegates are distinguishable');
  assert.match(lines[0].reply, /confident/, 'raw reply retained for the unsure case');
  assert.match(lines[1].reply, /turbo/, 'raw reply retained for the bad-label case');
  fs.rmSync(file, { force: true });
});
