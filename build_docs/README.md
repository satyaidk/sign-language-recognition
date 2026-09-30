# build_docs — The Project Operating Manual

> **Status (2026-09-30): this is the plan for the NEXT version, parked until the owner
> restarts it.** The current code is the refactored, tested `signlang` package
> (v0.2.0; see the root `README.md` and `CHANGELOG.md`). Where these files mention the
> old layout (`training/*.py`, root scripts, `reference/v0/`), map them with
> `docs/archive/original/README.md`. The v0.2 refactor already did part of phase P0:
> a package with a CLI, pinned requirements, and a pytest suite.

This folder tells any engineer or AI agent **what we are building, how it is built,
how work is done here, and what "finished" means**. It is the source of truth for
planning and status.

**What we're building:** **SignX v1**, a *language-agnostic* sign-language
recognition platform (data → landmarks → training → live recognition). It is
proven first on one sign language and designed to co-train several sign
languages later. The earlier 12-sign prototype (**v0**) is kept only as a
**reference**: its data is lost, and it's not retrained. Its technical
deep-dives (`docs/`, `training/docs/`, moving to `reference/v0/` in task T-0.1)
explain the proven patterns we port. The repo-root `CLAUDE.md` is the short entry
point that points here.

## Reading order (start → finish)

The files are numbered in the order a project is reasoned about: what the product
needs to do, what technology that requires, how it's structured, what rules apply,
how it will be built, what exactly is left to do, how it's verified, and where
things stand today.

| # | File | Answers | Read it when |
|---|------|---------|--------------|
| 01 | [01_PRD.md](01_PRD.md) | **What** should the product do, and for whom? What counts as done? | First time on the project, or when deciding scope |
| 02 | [02_TRD.md](02_TRD.md) | **Which technical capabilities** does that need? What's the gap today? | Before designing or changing a subsystem |
| 03 | [03_ARCHITECTURE.md](03_ARCHITECTURE.md) | **How is the software structured** (as-built and target)? What are the contracts? | Before touching code |
| 04 | [04_RULES.md](04_RULES.md) | **Which constraints** must every change respect? | **Every session** — non-negotiable |
| 05 | [05_IMPLEMENTATION_PLAN.md](05_IMPLEMENTATION_PLAN.md) | **How will we get there**: phases, order, exit criteria | When picking the next phase or re-planning |
| 06 | [06_TASKS.md](06_TASKS.md) | **What exactly** needs doing next, and the status of each task | **Every session** — pick work from here |
| 07 | [07_TEST.md](07_TEST.md) | **Which tests** we run, how to run them, and **the actual results** | Before and after every change |
| 08 | [08_MEMORY.md](08_MEMORY.md) | **Current state**, decisions made, gotchas, open questions, session log | **Every session** — read first, update last |

## Session protocol for agents (short version, full version in 04_RULES.md §1)

1. **Start:** read `08_MEMORY.md` → `06_TASKS.md` → `04_RULES.md`. Then open the
   PRD/TRD/ARCHITECTURE sections that the task you picked refers to.
2. **Pick** one task from `06_TASKS.md` whose dependencies are `DONE`. Mark it
   `IN-PROGRESS`.
3. **Work** within the rules. Run the tests listed in `07_TEST.md`.
4. **End:** update the task status and acceptance evidence in `06_TASKS.md`,
   append the test results to `07_TEST.md`, and add a session-log entry (plus
   any new decision or gotcha) to `08_MEMORY.md`.

## ID conventions used across these files

| Prefix | Meaning | Defined in |
|--------|---------|------------|
| `FR-n` | Functional requirement | 01_PRD |
| `NFR-n` | Non-functional requirement | 01_PRD |
| `TR-XXX-n` | Technical requirement / capability | 02_TRD |
| `ADR-n` | Architecture decision record | 03_ARCHITECTURE |
| `R-n` | Rule | 04_RULES |
| `P0…P7` | Implementation phase | 05_IMPLEMENTATION_PLAN |
| `T-p.n` | Task *n* of phase *p* | 06_TASKS |
| `UT/CT/DT/ST/RW-n` | Unit / component / data-model / system / real-world test | 07_TEST |
| `D-n` | Decision-log entry | 08_MEMORY |
| `Q-n` | Open question for the project owner | 08_MEMORY |

Anything marked **(proposed)** is a recommendation that the project owner has not
confirmed yet. See the open questions in `08_MEMORY.md`.
