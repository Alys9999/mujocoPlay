---
name: tech-design-doc-spec
description: Write or revise technical design documents so they conform to the repository's design-doc schema. Use when drafting, restructuring, reviewing, or upgrading architecture docs, protocol docs, scene integration docs, or cross-module infrastructure plans that should follow `docs/doc-schema-0222.md`.
---

# Tech Design Doc Spec

## Overview

Use this skill to turn an informal engineering memo into a review-ready technical design document. Keep the output decision-oriented, implementation-ready, and testable.

## Workflow

1. Read `../../docs/doc-schema-0222.md` when you need the full repository schema.
2. Classify the target document:
   - architecture upgrade
   - protocol design
   - scenario integration
   - cross-module infra refactor
3. Restructure the document into the repository's four-part design format.
4. Add missing evidence:
   - current-state analysis with code anchors
   - compatibility and migration strategy
   - implementation phases
   - tests and acceptance criteria
5. Add diagrams only where they clarify control flow, abstraction boundaries, or layering.
6. Keep terminology stable across the document. Do not rename the same concept in different sections.

## Required Output Shape

Produce a document that covers these semantic blocks:

- document scope and inputs
- executive summary
- part 1: problem definition and necessity
- part 2: current-state analysis
- part 3: solution details
- part 4: rollout plan and future work

Within those blocks, prefer including:

- design principles
- architecture overview
- capability matrix or staged roadmap
- compatibility and migration
- representative scenarios
- phased task table with test commands and acceptance criteria

## Hard Requirements

- Answer both "what problem is being solved" and "why now".
- Include both industry/framework context and repository-local current state.
- For repository-local claims, cite concrete file paths.
- When defining a protocol or schema, include:
  - field table
  - at least one minimal valid example
  - at least one full example
  - at least one failure example
  - error semantics and compatibility rules
- When proposing implementation work, provide phase, priority, status, acceptance criteria, test plan, and pytest command.

## Diagram Rules

Use Mermaid conservatively. Include the following diagram types when they materially help:

- `sequenceDiagram` for request / control / callback flow
- `flowchart` for branching process or lifecycle
- `classDiagram` for core abstractions and boundaries
- layered `flowchart` for architecture overview

Keep Mermaid compatible with simple node labels. Avoid complex punctuation in node ids.

## Writing Rules

- Prefer direct engineering language over ceremonial proposal language.
- Make decisions explicit.
- Separate current required scope from later expansion.
- Do not mix public policy-visible data and privileged/internal-only data in the same schema section.
- Keep sections actionable enough that they can become engineering tasks without extra interpretation.

## References

- Read [schema-summary.md](references/schema-summary.md) for the condensed checklist.
- Read `../../docs/doc-schema-0222.md` when you need the full source-of-truth wording.

## Output Check

Before finishing, verify:

- the document has a clear decision summary
- every major design claim has either code evidence or an explicit assumption
- phased tasks are independently testable
- test commands use `pytest`
- terminology is consistent end-to-end
