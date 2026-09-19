# jevdevice: full plan

A general tool that lets any agent operate any device. Three parts: a small local model supplies
small commands and turns messy output into structure; Jev judges at every junction; a graph of
proven commands is built while the system runs and makes later goals cheaper.

## 0. The goal and the model never meet (confirmed by the user 2026-09-19; overrides anything below)

- **Model**: asked only goal-free questions about how to run things on this device: how to list
  what can be run, how to get usage for one real entry, what invocations a piece of real help text
  documents. It never sees "open YouTube" or "CPU temp". It also turns messy real output into
  structure Jev can read.
- **Jev**: the only part that sees the goal. It generates nothing. It picks among real things
  (real output lines, documented invocations) and judges whether a step worked.
- **Graph**: written by the engine from what actually ran and which real output fed which command.
  The model never sees the graph and is never told what a node, slot or template looks like.
  Templates emerge from observed edges. Nobody authors them, the model included.

Why: given a goal, a 3B model writes the whole task from its priors (LOGBOOK: `svc radio set
bluetooth off`). Kept goal-blind, the answer can only come from the device's own real output.
Telling the model the shape of a template is hand-writing the templates one level up again.

## 1. The rule this plan is built around

Nobody hand-writes device commands. Not the user, not me, not in `src/`, not in scripts, not as
"probes" or "interface strings". Four earlier designs failed by removing one hardcoded thing and
hand-writing its replacement one level up (app table, action enum, six command templates, tuned
prompts, a static per-tool graph). Section 9 lists everything that IS hand-written so it can be
challenged line by line. Section 10 has the guards that stop me repeating the failure.

## 2. What the literature says (three review agents, 2026-09-19)

Verification status: [V] = I resolved the arXiv page myself or the agent fetched full text.
[A] = abstract or search snippet only; treat as a lead.

| Finding | Source | Consequence for us |
|---|---|---|
| Weak generator + strong separate verifier approaches a much larger model; verifier quality is the ceiling | Weaver 2506.18203 [V], Variation in Verification 2509.17995 [V] | The split is sound. Spend effort on what Jev is asked and shown, not on making the small model right first try |
| Sub-7B models are reliable when selecting from given options and hallucinate when inventing | Octopus v2 2404.01744 [A], Hammer 2410.04587 [A], our own FunctionGemma and qwen runs | Let the small model invent, but check what it invents against documentation it fetched, and let Jev pick |
| Documentation alone enables zero-shot tool use, better than demonstrations on large tool sets | Hsieh et al. 2308.00675 [V] | Fetching a tool's own help text is the cheapest reliability gain. Fetching it is itself a learned command, not a hand-written one |
| Repair loops plateau after 4 to 6 turns and then repeat mistakes | InterCode 2306.14898 [V] | Few parallel candidates plus a judge pick, repair capped at 2 turns |
| Judge success by diffing observable state before and after, not by reading the command or asking "does it look right" | NL2SH-ALFA 2502.06858 [V], SkillDroid 2604.14872 [V] | Verification = a state probe before and after, Jev judges the difference against the goal |
| Self-growing libraries collect garbage unless they track failures. Voyager and AWM never remove anything; AWM measured degraded performance and only 18.5% uptake of stored workflows | Voyager 2305.16291 [V], AWM 2409.07429 [V] | Per-node success stats, demotion, capped re-generation, from day one |
| SkillDroid (Android over adb): validates against device state not the model's claim, flags skills with >50% replay failure, caps recompilation at 3. 85.3% success, 49% fewer LLM calls, 100% over 79 replay rounds | SkillDroid 2604.14872 [V] | Closest prior system. Copy its invalidation scheme |
| Strong maker, cheap user: a capable model makes a tool once, a cheap model reuses it | LATM 2305.17126 [A] | When the small model cannot produce a working command, the outer agent supplies one; it passes the same gate and becomes a node |
| Hand-written command denylists fail to block what they intend in 69.0 to 98.6% of cases | Denylist Fragility 2606.15549 [V] | Do not rely on a prefix allowlist or denylist as the safety mechanism |
| Best alternative: static analysis first, judge only for unclear cases. No published scheme has zero human-written safety floor | CARE 2607.21642 [V] | A small human-written floor is unavoidable. Section 6 states exactly what it is |
| Best LLM judge reaches 74.42% at recognising risk from agent records | R-Judge 2401.10019 [A] | Never let one judge signal authorise a never-seen command. Require agreement of independent signals, else ask |
| Command output is an injection vector | InjecAgent 2403.02691 [A], AgentDojo 2406.13352 [A] | Output is only ever passed as a labelled data field. Anything derived from output re-passes the gate |
| The growing graph should be data interpreted by a small fixed engine; LangGraph's LLMCompiler tutorial does exactly this. Interrupts inside nested subgraphs can double-execute (GitHub #4796, #6792) | LangGraph docs [V by agent], LLMCompiler 2312.04511 [A] | Capability graph lives in the Store. Approval interrupt stays at top level |
| Edges can be inferred by matching one tool's output to another's input; naive matching over-connects | TaskBench, In-N-Out 2509.01560 [A] | An edge is only recorded after it worked once and Jev confirmed it |
| No published system grows a tool graph online from a small local model's trial and error. ControlLLM and HuggingGPT pre-build; LLMCompiler and ReWOO discard per query; AFlow, ADAS, GPTSwarm optimise offline | agent 2 survey | The parts have precedent; the whole does not. Milestones must measure, not assume |

## 3. Architecture

### 3.1 The capability graph (data, grows at runtime, stored in the LangGraph Store)

Namespace: device fingerprint (the fingerprint command is itself the first learned node).

**Node record** (one proven command):
- `goal_text`: the natural-language goal or sub-goal that produced it
- `command`: the literal string that ran, with `{slot}` holes for values that must come from the device
- `slots`: for each hole, a description and, once known, the producer node that supplies values
- `kind`: probe (judged read-only) or action
- `doc_excerpt`: the help text lines that support the command, when documentation was found
- `origin`: small_model, outer_agent, or repaired
- `stats`: runs, successes, consecutive failures, last verified, version
- `trust`: candidate, trusted, demoted

**Edge record**: "slot S of node A takes values from the output of node B", with a confirmation
count. Created only after the pairing worked in a real run and Jev confirmed the goal was met.

Nothing in the graph is authored. A node exists because a command was generated, passed the gate,
ran, and Jev judged from real state that it did what the goal asked.

### 3.2 The engine (small, fixed, knows nothing about any device, tool or command)

A LangGraph `StateGraph` with these generic steps. This is the only orchestration code.

```
recall      look up nodes for this goal in the Store (generic text narrowing), Jev Choice
            over the top matches plus "none fit"
propose     (only if none fit) small model emits up to 3 candidate commands, with {slots}
            for values it cannot know
ground      for a candidate's executable, run the sub-goal "show usage documentation for X"
            (itself recalled or learned), then small model revises the candidate against
            that text; Jev Choice picks the candidate most consistent with the documentation
classify    independent signals on whether the command changes state (section 6)
approve     top-level interrupt(); outer agent or user decides; resume continues
fill        each {slot} is a sub-goal "list values for <description>": recall or learn a probe,
            run it, narrow its output generically, Jev Choice over real lines, small model
            extracts the value from the chosen line, code checks it is a literal substring
observe     state probe before (a sub-goal: "show the state this goal is about")
execute     transport.run(command)
verify      state probe after, retried with backoff; Jev Noul on before/after versus the goal
record      update stats; promote, keep as candidate, or demote; write edges that were used
```

Routing uses conditional edges and `Command(goto=...)`. Sub-goals re-enter the same engine with a
depth cap. Budgets per goal: 3 candidates, 2 repair turns, sub-goal depth 2, a cap on Jev calls
and wall-clock. When the budget is spent the engine escalates to the outer agent with the trace.
The outer agent may supply a command. It goes through classify, approve, execute, verify like any
other and becomes a node with `origin: outer_agent`.

### 3.3 Who does what

| | Does | Never does |
|---|---|---|
| Small model (qwen2.5-coder:3b to start) | proposes candidate commands; revises them against fetched documentation; extracts a value from a line of output | picks between candidates, decides safety, decides success, sees a prompt tuned to a specific failure |
| Jev | recall match, candidate pick, slot value pick, state-change classification, gate, verification, edge confirmation. Every question is decidable and built by code from real data | reads raw unbounded output, generates anything |
| Code | transport, generic text narrowing, literal-substring grounding, budgets, stats, routing | contains a device command, an app name, a service name, a tool list |
| Outer agent (via MCP) | states goals, approves or rejects escalations, supplies a command when the small model is stuck | sits in the hot path for goals the graph already knows |

FunctionGemma has no role: selection is Jev's job and FunctionGemma cannot invent. It is dropped.

## 4. Handling large output

`dumpsys`-class output reaches 1 MB; Jev's state budget is 32k tokens. Code reduces output
generically before any model sees it: keep lines that fuzzy-match goal tokens, plus neighbours, up
to a size cap. The small model may also propose commands that filter at the source (it produced
`grep`/`awk` pipelines unprompted on the PC); those compete as ordinary candidates.

## 5. Judging success

- Action goals: the engine obtains a state probe for the goal (a learned node like any other),
  runs it before and after, and Jev answers one Noul: given before and after, was the goal
  achieved. If before and after are identical the action had no effect.
- Question goals: Jev Noul "does this output answer the question". The grounded excerpt is
  returned to the outer agent, which reads it. No per-goal checker is written.
- A probe whose output Jev judges irrelevant to the goal is discarded and another is proposed.

## 6. Safety (the honest version)

There is no published scheme with zero human-written safety floor. The floor here is:

1. **Fail closed.** A never-seen command runs without approval only if independent signals agree
   it does not change state: Jev Noul on the command text, Jev Noul on the command against its
   fetched documentation, and the small model's own label. Any disagreement, or no documentation,
   means approval is required.
2. **Approval goes to the outer agent first**, which knows shells far better than a 3B model, and
   to the user when the outer agent is unsure. Implemented with the top-level `interrupt()` that
   already passed a live test.
3. **Trust is earned.** An action node needs approval each run until it has k verified successes
   (k = 3 to start). Demoted nodes lose trust.
4. **Protect the link.** Before any action, Jev Noul: "could this command disable the connection
   this session uses", with the transport description as state. Yes or unsure means approval.
5. **Incident list.** A short list of literal patterns that caused real incidents, blocked
   outright. Today it has one entry family (the wifi disable that killed wireless adb). It grows
   only from incidents, never from imagination. It is a backstop, not the mechanism.
6. **Sandbox where one exists.** On the PC, never-seen commands run inside a read-only,
   no-network sandbox if one is available (checked in M0); otherwise rule 1 applies. adb has no
   sandbox, so the phone relies on rules 1 to 5.
7. **Output is data.** Output only enters prompts as a labelled field; anything it suggests goes
   through the same gate.

The existing `READ_ONLY_PREFIXES` allowlist and the general `DENY_SUBSTRINGS` list are removed.
The literature says such lists look safe and are not, and they are hand-written command knowledge.

## 7. Milestones

Each milestone ends with a written result in `LOGBOOK.md` carrying a `VERDICT:` line. No milestone
builds on an unverdicted one.

**M0. Ground and freeze (no engine code yet)**
- Repo cleanup: delete `toolkit.py`, the `discover_template` code, the two LangGraph test scripts
  move to `scripts/record/`. `prove_it.py`, `prove_toolkit.py`, `voice_prove_it.py`,
  `calibrate_gate.py` stay as the record of what was tested; they are not imported by `src/`.
- Write the goal sets as plain language, before any engine exists, and freeze them in
  `eval/goals.yaml`: 20 PC questions, 15 phone questions, 10 phone actions (never wifi or mobile
  data). Half of each set is marked held-out and is not looked at during development.
- Confirm from TypeSafe docs or one live call: the option cap for Choice and the state budget.
- Check whether a sandbox tool exists on the PC.
- `RESUME.md` and `LOGBOOK.md` created.
- Done when: goals frozen, limits recorded with their source, cleanup committed.

**M1. The engine on the PC, questions only**
- Build `engine.py` (section 3.2), `graph_store.py` (node and edge records over the Store),
  `smallmodel.py` (propose, revise, extract; prompts are fixed protocol text), `judge.py` (every
  Jev question in one place), `reduce.py` (section 4).
- Run the 10 development PC questions cold, then again warm.
- Report: first-candidate success, success within budget, Jev calls, small-model calls, latency,
  and on the warm pass the share answered with zero small-model calls.
- Done when: warm pass makes zero small-model calls on goals that succeeded cold, and the numbers
  are written down whatever they are.

**M2. Is this worth building (baseline comparison)**
- Baseline: the outer agent with one `run_shell` tool behind the same gate, same goals.
- Compare success, cost and latency, cold and warm. Spending estimate is stated and approved
  before the run.
- Decision written in the logbook. If the three-part loop does not clearly win warm on cost and
  latency at comparable success, stop and say so.

**M3. Slots and edges, phone questions and app launch**
- `{slot}` filling through sub-goals; edge records; demotion and the recompile cap.
- The phone's 8 development questions and "open <app>" goals, cold then warm.
- Done when: an "open <app>" goal succeeds with every token of the executed command traceable to
  the small model, fetched documentation, or real device output, and the warm pass reuses edges.

**M4. Actions on the phone**
- Section 6 in full; before/after verification; trust counters.
- The 5 development actions. Every never-seen action is approved by a person during this milestone.
- Done when: all safety rules have a test, and one deliberate bad candidate (supplied through the
  scripted model in a test) is stopped before execution.

**M5. MCP server**
- Tools: `device_do(goal)`, `device_ask(question)`, `device_approve(thread_id, decision,
  command?)`, `device_graph()` to inspect what has been learned.
- Registered in Claude Code and driven from a real session. Voice script rewired to call the
  same entry point.

**M6. Held-out evaluation**
- Run the held-out halves of all goal sets once, cold and warm. No changes between looking at
  results and reporting them. This is the number that says how general the system is.

## 8. Files

Keep as is: `transport.py`, `jev.py`, `matching.py`. Rework: `gate.py` becomes part of `judge.py`
with its recalibrated wording. Rewrite: `smallmodel.py`. Remove: `toolkit.py`, the prefix lists in
`grounding.py` (the literal-substring grounding check stays). New: `engine.py`, `graph_store.py`,
`judge.py`, `reduce.py`, `mcp_server.py`, `testing/` (FakeTransport replaying recorded outputs,
ScriptedJev, ScriptedSmallModel), `eval/`, `RESUME.md`, `LOGBOOK.md`.

## 9. Everything that is hand-written (complete list, challenge any line)

1. `Transport.run(command)` for adb and for a local shell. No command strings inside.
2. The engine steps in 3.2. They name no device, tool, app, service or command.
3. The fixed protocol prompts for propose, revise, extract. They describe the output format and
   the task in general terms only.
4. The wording of Jev's questions.
5. Budgets and thresholds.
6. The incident list in section 6, rule 5.
7. The goal sets in `eval/`, which are plain-language goals, not commands.

## 10. Guards against repeating the failure

- A test parses `src/` and fails if any call to `transport.run` receives a string literal or
  f-string. Commands may only arrive from a node record, a model candidate, or the outer agent.
- A test fails if `src/` contains any entry from a list of shell executables seen in traces.
- I do not run device commands by hand to "check" things. Evidence comes from engine traces.
- Prompts are versioned. A prompt or question wording changes only with a logbook entry, and the
  change must hold on goals it was not written for (the standard the HAL-path hint met on wifi
  and nfc). A failure is first handled by the loop and counted, not patched.
- Calibration facts kept from this session: decidable gate wording (0.82 to 0.94 vs 0.05); never
  take the first of a narrowed list, Jev picks; confidence below 0.6 or margin below 0.15
  escalates; verify with retry and backoff; never test a network-affecting action over the link it
  rides on; audible cue before any recording.

## 11. Decisions baked in (reject any of these and I will rework the plan)

1. The engine in 3.2 is fixed code; the graph that grows is data. The literature and LangGraph's
   own examples both point this way, and it keeps device knowledge out of code.
2. A human-written safety floor exists and is exactly section 6.
3. The PC comes first and questions come before actions, because mistakes there are cheapest.
4. M2 can end the project. I would rather find that out in a day.
5. When the small model is stuck, the outer agent may supply a command, and the graph learns it.

## 12. Verification of the whole

- `uv run pytest -q` passes with no hardware and no network, including the two guard tests.
- Live opt-in runs produce traces where every executed token is attributable.
- M6 numbers are reported as measured.
- `uv run ruff check src/` clean.
