# dsh-auto-answer

> A [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) plugin that answers `ask_user_question` prompts with an LLM, so option-style clarifying questions stop interrupting the operator.

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![node: >=20](https://img.shields.io/badge/node-%3E%3D20-brightgreen.svg)](package.json)

When an agent calls `ask_user_question`, DSH pauses the turn and shows a popup. This plugin steps in front of that popup and picks an option itself — but only when it is confident, and only for questions that actually offer options. Everything else falls through to the normal popup, unchanged.

## Why this is not `dsh-yolo-mode`

Both plugins auto-answer, but they sit on **different seams**:

| | `dsh-yolo-mode` | `dsh-auto-answer` (this) |
|---|---|---|
| Seam | `ctx.approval` → `approval/request` | `ctx.userQuestions` → `user-questions/request` |
| Trigger | a sandbox escalation (`escalate sandbox to <mode>: …`) | the `ask_user_question` tool |
| Default answerer | the host approval answerer | `dsh-api-remotes` → browser popup |
| When it fires | only from a mode strictly narrower than the target | in any mode, including `danger-full-access` |

A `danger-full-access` session produces **no** sandbox escalations at all: `WIDER_MODES` in `@deepseek-ai/dsh-sandbox` has no entry for `danger-full-access`, so an escalation throws before the approval seam is ever reached. The popups an operator sees in full access are `ask_user_question` — which is exactly what this plugin handles.

## Scope: option-only, and fail-closed

The plugin claims a request only when it can answer **every** question from that question's own option labels. Everything else calls `next()`, so the ordinary popup answers it:

- any question without `options` — free text needs facts only the operator has (a path, a secret, a personal preference, a business decision);
- no `judge.provider` / `judge.model` configured, or no `llm` service mounted;
- a verdict that is not `{"confident": true}`;
- a reply that is not JSON, or a label that is not one of that question's options;
- an unanswered question, or the wrong number of picks for a single-select question;
- timeout, abort, stream error, or concurrency overflow.

There is deliberately **no** "guess anyway" path. An answer invented on the operator's behalf is worse than one extra popup.

## How it hooks in

`ctx.userQuestions.ask()` runs a Cordis **waterfall** on `user-questions/request`, whose final fallback rejects with `NO_PROVIDER`. This plugin registers a listener with `{ prepend: true }`, so it runs before the browser answerer:

```js
ctx.effect(
  () => ctx.on('user-questions/request', handler, { prepend: true }),
  'auto-answer: user-questions answerer',
);
```

Returning `{ answers: [...] }` claims the request; returning `next()` leaves it to the human.

## Context

The decision uses the question text plus a truncated excerpt of the session's own `user/message` and `assistant/message` history (`judge.contextChars`, default 6000 characters, newest turns kept). Messages whose source is `agent-instructions` are skipped.

**That excerpt is the only session data sent to the model provider** — worth knowing when the provider is remote.

## Configuration

All fields are optional. Set them as the plugin row `config`:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `enabled` | `boolean` | `true` | `false` = never answer, always show the popup |
| `judge.provider` | `string` | `''` | LLM provider route; empty disables answering |
| `judge.model` | `string` | `''` | model id; empty disables answering |
| `judge.systemPrompt` | `string` | built-in | override the answering prompt |
| `judge.timeoutMs` | `number` | `60000` | per-request deadline |
| `judge.maxTokens` | `number` | `4096` | output budget (reasoning models want headroom) |
| `judge.concurrency` | `number` | `2` | in-flight answers; overflow delegates |
| `judge.contextChars` | `number` | `6000` | transcript budget sent to the model |
| `auditFile` | `string` | `~/.dsh/logs/auto-answer.jsonl` | JSONL trail |

Example:

```yaml
- id: auto-answer
  name: dsh-auto-answer
  config:
    enabled: true
    judge:
      provider: my-provider
      model: my-model
      timeoutMs: 60000
      maxTokens: 4096
      contextChars: 6000
    auditFile: /home/me/.dsh/logs/auto-answer.jsonl
```

## Install

```bash
dsh plugin --profile web add /path/to/dsh-auto-answer
```

The bundle's `cordis.patch.yml` inserts the `auto-answer` entry; override its `config` by id in the profile's `cordis.patch.yml` (do **not** re-insert it — that fails with `duplicate loader entry id`).

For a path or `link:` install, keep the checkout under `$DSH_HOME/profiles/` so Node resolves the `@deepseek-ai/*` peers from `$DSH_HOME/profiles/node_modules`.

A **new** plugin module only mounts on a DSH restart; afterwards config edits reload live under `patchReload: live`.

## Audit trail

One JSONL line per request, at `auditFile`:

```json
{"time":1790422374944,"sessionId":"session-…","questionIds":["preset"],"outcome":"answered","answers":[{"id":"preset","selected":["permissive"]}]}
{"time":1790422375001,"sessionId":"session-…","questionIds":["key"],"outcome":"delegate","error":"BAD_OUTPUT","message":"reply did not answer every question with valid option labels"}
```

`outcome` is `answered` or `delegate`; `delegate` carries the `error` code that sent the request to the human (`BAD_OUTPUT`, `TIMEOUT`, `STREAM_ERROR`, `NO_ADAPTER`, `OVERLOAD`).

## Tests

```bash
npm install
npm test
```

30 tests in two layers:

- `test/logic.test.mjs` — config normalization, tolerant JSON extraction, the strict option-label validator, transcript building.
- `test/probe.test.mjs` — mounts the plugin on a **real Cordis context** and drives the actual `user-questions/request` waterfall. It asserts both halves of the contract: a confident verdict claims the request and the human answerer never runs, and every doubtful path delegates to it.

The `devDependencies` exist only so the tests run from a bare clone; at runtime the plugin resolves its `@deepseek-ai` peers from the DSH install. The suite is verified against both `0.1.5-rc.2` (the version shipped inside DSH 0.1.5) and the published `0.1.7-rc.2` peers.

## License

MIT — see [LICENSE](LICENSE).
