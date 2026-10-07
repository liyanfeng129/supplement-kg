# Coding standards

Rules a reviewer judges against a diff. Rules a script can check (import
direction, folder names, modules outside a layer) live in
`tests/test_layers.py`, not here.

1. **Write only the functions the design agreed.** A spec or ADR names the
   functions a change adds; each does one thing and its logic lives inside it,
   even when that makes it long, with comments marking the steps. A private
   helper, helper class or module-level utility that no design named is a
   finding, including one extracted to remove duplication. Tests patch library
   boundaries (`subprocess.Popen`, `urllib.request.urlopen`, `time.monotonic`)
   rather than getting private methods made as seams.

2. **Each function sits in the layer that owns its work.** The owner is the
   "Owns" column of the table in the latest
   `docs/adr/0002-engine-architecture/vN.md` §3. Code that does a lower
   layer's work from a higher one (a service building a model, a controller
   reading files) or that collects several layers' work in one place (one
   loop that queues, persists, retries and reports) is a finding.

3. **A schema lives with its producer.** A type is defined beside the loader,
   model or service that creates it. A shared `schemas/` module is a finding.

4. **A change that contradicts the latest ADR version adds the next version.**
   See "ADR versions" in `docs/agents/domain.md`. Code that contradicts an
   ADR without a new `vN+1.md` in the same change is a finding.
