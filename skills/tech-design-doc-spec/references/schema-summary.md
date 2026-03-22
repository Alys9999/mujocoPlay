# Schema Summary

Source of truth: `../../docs/doc-schema-0222.md`

Use this summary when you need the checklist quickly and do not need the full wording.

## Required top-level structure

Every technical design document should cover these semantic blocks:

1. Document scope and input materials
2. Executive summary
3. Part 1: problem definition and necessity
4. Part 2: current-state analysis
5. Part 3: solution details
6. Part 4: rollout plan and future work

## Strongly recommended sections

- design principles
- architecture overview
- compatibility and migration
- capability matrix or staged roadmap
- representative scenarios
- phased implementation tasks

## Evidence requirements

- Include both industry/framework context and repo-local current state.
- For repo-local claims, cite concrete file paths.
- State assumptions explicitly when there is no direct evidence.

## Diagram requirements

Use Mermaid only when it clarifies something important. Prefer including:

- `sequenceDiagram`
- `flowchart`
- `classDiagram`
- layered architecture diagram

Keep node labels simple for Mermaid compatibility.

## Protocol/schema sections

If the document defines a protocol, schema, or event format, include:

- field table
- minimal valid example
- full example
- failure example
- error semantics
- compatibility rules

## Implementation plan requirements

Implementation tasks should be independently testable. Each task row should include:

- sequence id
- phase
- priority
- status
- task
- acceptance criteria
- test plan
- pytest command
- code scope
- dependencies
- risks

## Quality gate

Before finalizing, check:

1. The document explains why this work is necessary now.
2. Major design claims have evidence or clearly marked assumptions.
3. Terminology is stable across the whole document.
4. Current required scope and later expansion are clearly separated.
5. Test strategy is concrete and executable.
