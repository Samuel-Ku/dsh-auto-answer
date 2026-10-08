/**
 * Pure-logic tests for dsh-auto-answer: config normalization, tolerant JSON
 * extraction, the option-label resolver, the verdict validator, and transcript
 * building.
 *
 * Run: node --test test/
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_SYSTEM_PROMPT,
  normalizeConfig,
  parseJsonObject,
  resolveLabel,
  transcriptOf,
  validateVerdict,
} from '../lib/index.js';

const QUESTIONS = [
  {
    id: 'preset',
    question: 'Which preset?',
    options: [{ label: 'balanced (Recommended)' }, { label: 'permissive' }, { label: 'yolo' }],
  },
];

/* ------------------------------------------------------------------ config */

test('normalizeConfig: defaults are inert-but-sane', () => {
  const c = normalizeConfig(undefined);
  assert.equal(c.enabled, true);
  assert.equal(c.judge.provider, '');
  assert.equal(c.judge.model, '');
  assert.equal(c.judge.systemPrompt, DEFAULT_SYSTEM_PROMPT);
  assert.equal(c.judge.timeoutMs, 60000);
  assert.equal(c.judge.maxTokens, 4096);
  assert.equal(c.judge.concurrency, 2);
  assert.equal(c.judge.repairAttempts, 1, 'one repair turn by default');
  assert.match(c.auditFile, /auto-answer\.jsonl$/);
});

test('normalizeConfig: repairAttempts accepts zero but nothing below it', () => {
  assert.equal(normalizeConfig({}).judge.repairAttempts, 1);
  assert.equal(normalizeConfig({ judge: { repairAttempts: 0 } }).judge.repairAttempts, 0, '0 is the off switch');
  assert.equal(normalizeConfig({ judge: { repairAttempts: -1 } }).judge.repairAttempts, 1);
  assert.equal(normalizeConfig({ judge: { repairAttempts: 2.5 } }).judge.repairAttempts, 1);
  assert.equal(normalizeConfig({ judge: { repairAttempts: '3' } }).judge.repairAttempts, 1);
});

test('normalizeConfig: garbage degrades to defaults, never throws', () => {
  const c = normalizeConfig({ enabled: 'no', judge: { timeoutMs: -5, maxTokens: 'x', concurrency: 0 } });
  assert.equal(c.enabled, true, 'only an explicit false disables');
  assert.equal(c.judge.timeoutMs, 60000);
  assert.equal(c.judge.maxTokens, 4096);
  assert.equal(c.judge.concurrency, 2);
});

test('normalizeConfig: explicit values win', () => {
  const c = normalizeConfig({
    enabled: false,
    judge: { provider: 'p', model: 'm', systemPrompt: 'S', timeoutMs: 1000, maxTokens: 512, concurrency: 4, contextChars: 100 },
    auditFile: '/tmp/x.jsonl',
  });
  assert.equal(c.enabled, false);
  assert.equal(c.judge.provider, 'p');
  assert.equal(c.judge.model, 'm');
  assert.equal(c.judge.systemPrompt, 'S');
  assert.equal(c.judge.timeoutMs, 1000);
  assert.equal(c.judge.maxTokens, 512);
  assert.equal(c.judge.concurrency, 4);
  assert.equal(c.judge.contextChars, 100);
  assert.equal(c.auditFile, '/tmp/x.jsonl');
});

/* ------------------------------------------------------- parseJsonObject */

test('parseJsonObject: plain, fenced and prose-wrapped replies', () => {
  assert.deepEqual(parseJsonObject('{"a":1}'), { a: 1 });
  assert.deepEqual(parseJsonObject('```json\n{"a":1}\n```'), { a: 1 });
  assert.deepEqual(parseJsonObject('Here you go:\n{"a":{"b":[1,2]}}\nDone.'), { a: { b: [1, 2] } });
});

test('parseJsonObject: braces inside strings do not confuse the scan', () => {
  assert.deepEqual(parseJsonObject('{"reason":"a } b { c"}'), { reason: 'a } b { c' });
});

test('parseJsonObject: rejects non-objects and malformed input', () => {
  assert.equal(parseJsonObject('[1,2]'), null);
  assert.equal(parseJsonObject('{"a":'), null);
  assert.equal(parseJsonObject('no json here'), null);
  assert.equal(parseJsonObject(''), null);
  assert.equal(parseJsonObject(null), null);
});

/* ------------------------------------------------------- validateVerdict */

test('validateVerdict: a well-formed confident verdict passes', () => {
  const v = validateVerdict({ answers: [{ id: 'preset', selected: ['permissive'] }], confident: true }, QUESTIONS);
  assert.deepEqual(v, { answers: [{ id: 'preset', selected: ['permissive'] }] });
});

test('validateVerdict: label match is case/space tolerant but canonicalizes', () => {
  const v = validateVerdict({ answers: [{ id: 'preset', selected: ['  YOLO '] }], confident: true }, QUESTIONS);
  assert.deepEqual(v.answers[0].selected, ['yolo'], 'returns the exact option label');
});

test('validateVerdict: fails closed on every doubtful shape', () => {
  const bad = [
    [{ answers: [{ id: 'preset', selected: ['yolo'] }] }, 'missing confident flag'],
    [{ answers: [{ id: 'preset', selected: ['yolo'] }], confident: false }, 'not confident'],
    [{ answers: [{ id: 'preset', selected: ['invented option'] }], confident: true }, 'unknown label'],
    [{ answers: [{ id: 'other', selected: ['yolo'] }], confident: true }, 'missing question id'],
    [{ answers: [{ id: 'preset', selected: [] }], confident: true }, 'empty selection'],
    [{ answers: [{ id: 'preset', selected: ['yolo', 'permissive'] }], confident: true }, 'two picks for single-select'],
    [{ answers: [{ id: 'preset', selected: ['yolo'] }, { id: 'preset', selected: ['permissive'] }], confident: true }, 'duplicate id'],
    [{ answers: [{ id: 'preset', selected: ['yolo', 'yolo'] }], confident: true }, 'collapses duplicate to one (allowed)'],
    [{ answers: 'nope', confident: true }, 'answers not an array'],
    [null, 'null'],
  ];
  for (const [verdict, label] of bad) {
    const got = validateVerdict(verdict, QUESTIONS);
    if (label === 'collapses duplicate to one (allowed)') {
      assert.deepEqual(got, { answers: [{ id: 'preset', selected: ['yolo'] }] }, label);
    } else {
      assert.equal(got, null, label);
    }
  }
});

test('validateVerdict: multiSelect allows and preserves several picks', () => {
  const qs = [{ id: 'm', question: 'pick', multiSelect: true, options: [{ label: 'a' }, { label: 'b' }, { label: 'c' }] }];
  const v = validateVerdict({ answers: [{ id: 'm', selected: ['c', 'a'] }], confident: true }, qs);
  assert.deepEqual(v.answers[0].selected, ['c', 'a']);
  assert.equal(validateVerdict({ answers: [{ id: 'm', selected: [] }], confident: true }, qs), null);
});

test('validateVerdict: every question must be answered', () => {
  const qs = [
    { id: 'a', question: 'A', options: [{ label: 'x' }] },
    { id: 'b', question: 'B', options: [{ label: 'y' }] },
  ];
  assert.equal(validateVerdict({ answers: [{ id: 'a', selected: ['x'] }], confident: true }, qs), null);
  assert.deepEqual(
    validateVerdict({ answers: [{ id: 'b', selected: ['y'] }, { id: 'a', selected: ['x'] }], confident: true }, qs).answers.map((x) => x.id),
    ['a', 'b'],
    'returned in the asked order',
  );
});

/* ---------------------------------------------------------- transcriptOf */

function fakeAgent(events) {
  return { session: { id: 'session-x', events } };
}

test('transcriptOf: renders roles in order and skips non-message events', () => {
  const agent = fakeAgent([
    { type: 'user/message', data: { content: [{ type: 'text', text: 'hello' }] } },
    { type: 'tool/call', data: { toolName: 'bash' } },
    { type: 'assistant/message', data: { message: { content: [{ type: 'text', text: 'hi there' }] } } },
  ]);
  const out = transcriptOf(agent, 6000);
  assert.match(out, /\[operator\] hello/);
  assert.match(out, /\[assistant\] hi there/);
  assert.ok(out.indexOf('hello') < out.indexOf('hi there'), 'chronological');
});

test('transcriptOf: skips agent-instructions user messages', () => {
  const agent = fakeAgent([
    { type: 'user/message', data: { source: { kind: 'agent-instructions' }, content: 'SECRET INSTRUCTIONS' } },
    { type: 'user/message', data: { content: 'real question' } },
  ]);
  const out = transcriptOf(agent, 6000);
  assert.ok(!out.includes('SECRET INSTRUCTIONS'));
  assert.match(out, /real question/);
});

test('transcriptOf: budget keeps the most recent turns', () => {
  const events = [];
  for (let i = 0; i < 20; i++) events.push({ type: 'user/message', data: { content: `message-${i}-${'x'.repeat(50)}` } });
  const out = transcriptOf(fakeAgent(events), 200);
  assert.ok(out.includes('message-19'), 'newest kept');
  assert.ok(!out.includes('message-0-'), 'oldest dropped');
});

test('transcriptOf: survives a missing/odd agent', () => {
  assert.equal(transcriptOf(undefined, 100), '');
  assert.equal(transcriptOf({}, 100), '');
  assert.equal(transcriptOf({ session: {} }, 100), '');
  assert.equal(transcriptOf(fakeAgent([]), 100), '');
});

/* ----------------------------------------------------------- resolveLabel */

const DECORATED = [
  'Перевантажити метод і зберегти сумісність (Recommended)',
  'Змінити сигнатуру й оновити всі виклики',
  'Винести в окремий метод',
];

test('resolveLabel: exact and whitespace/case-normalized matches', () => {
  assert.equal(resolveLabel(DECORATED[1], DECORATED), DECORATED[1]);
  assert.equal(resolveLabel('  змінити   СИГНАТУРУ й оновити всі виклики ', DECORATED), DECORATED[1]);
});

test('resolveLabel: a dropped trailing decoration still resolves', () => {
  assert.equal(resolveLabel('Перевантажити метод і зберегти сумісність', DECORATED), DECORATED[0]);
  assert.equal(resolveLabel('Перевантажити метод і зберегти сумісність (Рекомендовано)', DECORATED), DECORATED[0]);
});

test('resolveLabel: a unique prefix in either direction resolves', () => {
  assert.equal(resolveLabel('Винести в окремий', DECORATED), DECORATED[2], 'pick is a prefix of the label');
  assert.equal(resolveLabel('Винести в окремий метод, бо так чистіше', DECORATED), DECORATED[2], 'label is a prefix of the pick');
});

test('resolveLabel: refuses the moment two options could both match', () => {
  const sameStem = ['Варіант A (Recommended)', 'Варіант A (Alternative)'];
  assert.equal(resolveLabel('Варіант A (Recommended)', sameStem), 'Варіант A (Recommended)', 'an exact match wins outright');
  assert.equal(resolveLabel('Варіант A', sameStem), null, 'the bare stem names both options once decorations are stripped');
  assert.equal(resolveLabel('Варіант', sameStem), null, 'a prefix of both');
});

test('resolveLabel: a case variant still resolves when only one option fits', () => {
  const labels = ['Варіант A (Recommended)', 'Варіант B'];
  assert.equal(resolveLabel('варіант a', labels), 'Варіант A (Recommended)');
});

test('resolveLabel: refuses a short or foreign pick', () => {
  assert.equal(resolveLabel('так', DECORATED), null, 'too short to prefix-match safely');
  assert.equal(resolveLabel('Вигаданий варіант', DECORATED), null);
  assert.equal(resolveLabel('', DECORATED), null);
});

test('validateVerdict: uses the tolerant resolver but still needs a real option', () => {
  const qs = [{ id: 'q', question: 'Що робити?', options: DECORATED.map((label) => ({ label })) }];
  const ok = validateVerdict({ answers: [{ id: 'q', selected: ['Винести в окремий метод'] }], confident: true }, qs);
  assert.deepEqual(ok.answers[0].selected, [DECORATED[2]], 'canonical label returned');
  assert.equal(
    validateVerdict({ answers: [{ id: 'q', selected: ['Щось інше'] }], confident: true }, qs),
    null,
  );
});
