"""What is left of the first local extraction pipeline for DSLD purpose/risk.

The orchestrator (`run`, `drivers`, `cli`, `compare`) is gone: a purpose/risk
run is now `services/purpose_risk/service.py`. These stages stay because
committed work still imports them: `render` (the renderer the pilot passages
and the gold and silver sets were built with, used by `datasets/pilot/`),
`extract` and `config` (`datasets/pilot/compare_gold1000_inputs.py`), `lines`
and `validate` (`benchmarks/parallel_slots.py`), and `score`. See
docs/adr/0001-local-llm-purpose-risk-pipeline.md.
"""
