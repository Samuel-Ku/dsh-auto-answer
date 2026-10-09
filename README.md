# dsh-auto-answer

> A [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) plugin that answers `ask_user_question` prompts with an LLM, so option-style clarifying questions stop interrupting the operator.

[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![node: >=20](https://img.shields.io/badge/node-%3E%3D20-brightgreen.svg)](package.json)

When an agent calls `ask_user_question`, DSH pauses the turn and shows a popup. This plugin steps in front of that popup and picks an option itself — but only when it is confident, only for questions that actually offer options, and only after a malformed or mislabelled first reply has had its repair turn. When it still cannot settle the question, a second-opinion model with more of the conversation gets a try, and only then does the popup appear — where the wait is bounded by `fallback.afterMs` instead of lasting until you wake up.

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
- a reply that is not JSON, or a label that names no option of that question;
- an unanswered question, or the wrong number of picks for a single-select question;
- timeout, abort, stream error, or concurrency overflow.

There is deliberately **no** "guess anyway" path. An answer invented on the operator's behalf is worse than one extra popup.

### A second opinion before the operator is bothered

A `NOT_CONFIDENT` verdict from the primary judge is not the end of the road. The
`escalation.*` profile — typically a bigger model given four times the transcript
budget — is asked the same questions with one instruction the primary lacks: the
first judge already refused, so this one is told to commit. Its built-in prompt
states what outranks what (the operator's own words above every agent report,
this project's written decisions and its SDLC above improvisation, reversibility
as the tie-breaker), which is what turns "not confident" into a defensible
answer.

### The wait for the operator is bounded

Once a question is genuinely the operator's, the popup opens — but the plugin
races it against `fallback.afterMs` (default 10 minutes, `0` disables). On
timeout it asks the escalation profile to choose *now*; if that fails too, it
takes the single option marked `(Recommended)`, or the only option there is. A
question that offers a real choice and defeats both judges keeps waiting, and the
audit line says so. Without this, one unanswered popup stopped a session for
15 hours.

### A malformed reply gets one repair turn

The first reply is not always the final word: a model that narrates instead of
emitting JSON, or paraphrases a label, is told what it did wrong and asked again
with the rejected reply quoted back. `judge.repairAttempts` (default `1`, `0`
disables) bounds those extra turns, and they share the request's single deadline
— a repair can never extend the wait past `judge.timeoutMs`. A request that
exhausts its repairs fails with the last error, exactly as before. The audit line
records every rejected reply under `repairs`.

### Option labels are matched leniently — but only about formatting

Asking agents decorate labels (`balanced (Recommended)`, `glm-5.3-flash (2x usage)`), and a model that echoes only the stem used to be treated as inventing an option, losing a perfectly answerable question to a popup. `resolveLabel` therefore widens in four stages — exact, whitespace/case-normalized, trailing-parenthetical stripped, then unique prefix in either direction — and **every stage demands a unique hit**. The moment two options could both match, it returns null and the question goes to the human. Tolerance is about formatting only; option *identity* is never guessed.

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
| `judge.maxMessageChars` | `number` | `1200` | per-message cap, so one long report cannot crowd out the rest |
| `judge.repairAttempts` | `number` | `1` | extra turns for a malformed or mislabelled reply before delegating; `0` = never repair |
| `escalation.provider` | `string` | `''` | second-opinion provider; empty disables the second opinion |
| `escalation.model` | `string` | `''` | second-opinion model id |
| `escalation.systemPrompt` | `string` | built-in | override the second-opinion prompt |
| `escalation.timeoutMs` | `number` | `120000` | per-call deadline of the second opinion (its own, not shared) |
| `escalation.maxTokens` | `number` | `8192` | output budget of the second opinion |
| `escalation.contextChars` | `number` | `24000` | transcript budget of the second opinion |
| `fallback.afterMs` | `number` | `600000` | how long the popup may block before the fallback answers; `0` = wait for ever |
| `fallback.allowRecommended` | `boolean` | `true` | allow the last-resort `(Recommended)` pick when no model can decide |
| `auditFile` | `string` | `~/.dsh/logs/auto-answer.jsonl` | JSONL trail |

Example:

```yaml
- id: auto-answer
  name: dsh-auto-answer
  config:
    enabled: true
    judge:
      provider: my-provider
      model: my-fast-model
      timeoutMs: 60000
      maxTokens: 4096
      contextChars: 6000
      maxMessageChars: 1200
      repairAttempts: 1
    escalation:
      provider: my-provider
      model: my-big-model
      contextChars: 24000
      maxTokens: 8192
      timeoutMs: 120000
    fallback:
      afterMs: 600000
      allowRecommended: true
    auditFile: /home/me/.dsh/logs/auto-answer.jsonl
```

## Install

```bash
dsh plugin --profile web add /path/to/dsh-auto-answer
```

The bundle's `cordis.patch.yml` inserts the `auto-answer` entry; override its `config` by id in the profile's `cordis.patch.yml` (do **not** re-insert it — that fails with `duplicate loader entry id`).

For a path or `link:` install, keep the checkout under `$DSH_HOME/profiles/` so Node resolves the `@deepseek-ai/*` peers from `$DSH_HOME/profiles/node_modules`.

A **new** plugin module only mounts on a DSH restart — and so does **updated
plugin code**. `patchReload: live` re-applies `config`; it does not re-import a
module the host already holds in memory, so a pull that changes `lib/` keeps
running the old code until dsh restarts. Config edits reload live.

## Audit trail

One JSONL line per request, at `auditFile`:

```json
{"time":1790422374944,"sessionId":"session-…","questionIds":["preset"],"outcome":"answered","via":"judge","answers":[{"id":"preset","selected":["permissive"]}]}
{"time":1790422375001,"sessionId":"session-…","questionIds":["key"],"outcome":"delegate","error":"NOT_CONFIDENT","message":"the model judged the excerpt insufficient to answer","reply":"{\"confident\":false}"}
{"time":1790422375100,"sessionId":"session-…","questionIds":["preset"],"outcome":"answered","via":"escalation","answers":[{"id":"preset","selected":["balanced (Recommended)"]}],"trace":[{"attempt":"escalation","error":"NOT_CONFIDENT","message":"the model judged the excerpt insufficient to answer","reply":"{\"confident\":false}"}]}
{"time":1790422375200,"sessionId":"session-…","questionIds":["preset"],"outcome":"answered","via":"fallback-escalation","answers":[{"id":"preset","selected":["yolo"]}],"fallback":{"afterMs":600000}}
```

`via` names what actually decided: `judge`, `escalation`, `fallback-escalation`,
`fallback-judge`, or `fallback-recommended`. `trace` appears only when a reply was
rejected, repaired or escalated — it carries the same `error` / `message` /
`reply` shape as a delegate, plus `attempt: "escalation"` for the hand-off.
`fallback` records that the operator did not answer in time, and
`fallback.forcedError` explains a fallback that had to fall back further.

`outcome` is `answered` or `delegate`. A delegate always names why:

| `error` | Meaning | Actionable? |
|---|---|---|
| `NOT_CONFIDENT` | the model judged the excerpt insufficient — and the second opinion agreed, or none is configured | working as designed — give it more context, point `escalation.*` at a bigger model, or answer by hand |
| `BAD_LABELS` | the reply did not line up with the offered options, after the repair turn | worth reading `reply`; usually a model that paraphrased |
| `BAD_OUTPUT` | the reply was not a JSON object, was empty, or contained a tool call, after the repair turn | often a truncated reply — raise `judge.maxTokens` |
| `TIMEOUT` | the request outlived `judge.timeoutMs` | raise the timeout, or check for a plugin parking `llm/stream` |
| `STREAM_ERROR` | the provider stream threw | check provider health |
| `NO_ADAPTER` | no provider/model configured | configure them |
| `OVERLOAD` | more concurrent questions than `judge.concurrency` | raise it |
| `ABORTED` | the turn was cancelled while answering | — |

When the model's *own text* caused the failure (`BAD_OUTPUT`, `BAD_LABELS`, `NOT_CONFIDENT`), the entry also carries a `reply` excerpt — otherwise a delegate cannot be explained after the fact.

## Tests

```bash
npm install
npm test
```

58 tests in two layers:

- `test/logic.test.mjs` — config normalization (including `escalation.*` and `fallback.*`), tolerant JSON extraction, the strict option-label validator, transcript classification and capping, the `(Recommended)` last resort, transcript building.
- `test/probe.test.mjs` — mounts the plugin on a **real Cordis context** and drives the actual `user-questions/request` waterfall. It asserts both halves of the contract: a confident verdict claims the request and the human answerer never runs, and every doubtful path delegates to it. Eleven of them cover the newer paths: the repair pass, the second opinion claiming an unsure question with a bigger excerpt, a still-unsure second opinion falling through, the bounded wait answering while the popup stays open, the `(Recommended)` last resort, a prompt operator answer never being overridden, and `fallback.afterMs: 0` keeping the old unbounded wait.

The `devDependencies` exist only so the tests run from a bare clone; at runtime the plugin resolves its `@deepseek-ai` peers from the DSH install. The suite is verified against both `0.1.5-rc.2` (the version shipped inside DSH 0.1.5) and the published `0.1.7-rc.2` peers.

## License

MIT — see [LICENSE](LICENSE).
