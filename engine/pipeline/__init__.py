"""Local extraction pipeline for DSLD purpose/risk.

Six modules, one per stage: `render` (label to passage), `extract` (one record
to one result), `validate` (raw to validated rows), `score` (rows plus gold to
metrics), `run` (the orchestrator), `cli` (argparse only).

Three drivers -- `local`, `api`, `agent` -- write the same row shape. The
enforced abstraction is that row plus `validate()`, not a Backend class:
subagents can only be spawned from inside an agent's turn, so a uniform
interface would have to lie about one of its implementations. `provenance` is
a column, never a branch. See docs/adr/0001-local-llm-purpose-risk-pipeline.md.
"""
