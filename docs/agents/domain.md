# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`GLOSSARY.md`** at the repo root, or
- **`GLOSSARY-MAP.md`** at the repo root if it exists: it points at one `GLOSSARY.md` per context. Read each one relevant to the topic.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in, the latest version of each (see [ADR versions](#adr-versions)). In multi-context repos, also check `src/<context>/docs/adr/` for context-scoped decisions.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

Single-context repo (most repos):

```
/
├── GLOSSARY.md
├── docs/adr/
│   ├── 0001-event-sourced-orders/
│   │   ├── v1.md
│   │   └── v2.md
│   └── 0002-postgres-for-write-model/
│       └── v1.md
└── src/
```

Multi-context repo (presence of `GLOSSARY-MAP.md` at the root):

```
/
├── GLOSSARY-MAP.md
├── docs/adr/                          ← system-wide decisions
└── src/
    ├── ordering/
    │   ├── GLOSSARY.md
    │   └── docs/adr/                  ← context-specific decisions
    └── billing/
        ├── GLOSSARY.md
        └── docs/adr/
```

## ADR versions

This repo versions ADRs instead of superseding them in place:

- **One folder per ADR, one file per version**: `docs/adr/NNNN-<slug>/vN.md`. The highest `vN.md` is the decision in force; earlier versions are the record and are never edited.
- **The latest version describes the code as it is.** A design not yet built lives in `docs/specs/`; a new ADR version is written when the code lands.
- **A change that contradicts the latest version includes `vN+1.md` in the same change.** Each new version says at its top what changed from the one before.
- A new ADR starts as `NNNN-<slug>/v1.md`, not a flat `NNNN-<slug>.md`.
- Link to an ADR by its folder (`docs/adr/0002-engine-architecture/`), so the link survives new versions.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `GLOSSARY.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
