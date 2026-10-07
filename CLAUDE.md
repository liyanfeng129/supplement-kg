## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (liyanfeng129/supplement-kg), via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: root `GLOSSARY.md` + `docs/adr/`. See `docs/agents/domain.md`.

## Code structure

Adding or moving a module under `engine/`: read the latest version in `docs/adr/0002-engine-architecture/` for what each layer owns.
