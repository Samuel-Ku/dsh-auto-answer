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
    judge: { provider: 'openrouter-jev', model: 'typesafe/jev-router', timeoutMs: 5000, maxTokens: 512, concurrency: 2 },
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
  assert.equal(llm.calls[0].provider, 'openrouter-jev');
  assert.equal(llm.calls[0].model, 'typesafe/jev-router');
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
 * audit trail
 * ------------------------------------------------------------------ */

test('probe: writes an audit line for both outcomes', async () => {
  const file = auditPath();
  const confident = await boot({ ...baseConfig(), auditFile: file }, () =>
    fromChunks(textChunks('{"answers":[{"id":"preset","selected":["yolo"]}],"confident":true,"reason":"ок"}')),
  );
  await confident.ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);

  const doubtful = await boot({ ...baseConfig(), auditFile: file }, () => fromChunks(textChunks('nope')));
  await doubtful.ctx.waterfall('user-questions/request', request(QUESTIONS), humanFallback);

  const lines = await readAuditLines(file, 2);
  assert.deepEqual(lines.map((l) => l.outcome), ['answered', 'delegate']);
  assert.equal(lines[1].error, 'BAD_OUTPUT');
  assert.deepEqual(lines[0].answers, [{ id: 'preset', selected: ['yolo'] }]);
  fs.rmSync(file, { force: true });
});
