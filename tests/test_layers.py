"""engine/ layers: nothing imports upward, and no folder is named for a sequence step.

What is pinned (ADR 0002, latest version): a module imports only its own layer
or a layer below it, in the order pipeline > services > controllers >
repositories > models; `engine/config.py` and `engine/notify.py` are shared
by every layer and import none; every module under `engine/` sits in a layer
or is one of those shared modules; and no folder is called `preprocessing`,
`processing` or `postprocessing`.

Reads source with `ast`; imports nothing from `engine`, so it needs no GPU and
no data.
"""

from __future__ import annotations

import ast
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent / "engine"

# Higher may import lower or equal; never the other way.
RANK = {"pipeline": 4, "services": 3, "controllers": 2, "repositories": 1, "models": 0}
SHARED = {"config", "notify"}
SEQUENCE_NAMES = {"preprocessing", "processing", "postprocessing", "pre_processing", "post_processing"}


def test_no_module_imports_a_layer_above_its_own():
    violations = []
    for path in sorted(ENGINE.rglob("*.py")):
        parts = path.relative_to(ENGINE).with_suffix("").parts
        top = parts[0]
        if top == "__init__":
            continue
        source_rank = RANK.get(top)  # None for a shared module

        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                # `from engine import config` names the module in the alias.
                if node.module == "engine":
                    targets = [f"engine.{alias.name}" for alias in node.names]
                else:
                    targets = [node.module]
            else:
                continue

            for target in targets:
                target_parts = target.split(".")
                if target_parts[0] != "engine" or len(target_parts) < 2:
                    continue
                target_top = target_parts[1]
                if target_top in SHARED or target_top == top:
                    continue
                target_rank = RANK.get(target_top)
                if target_rank is None:
                    continue  # an unknown top-level name is reported by the next test
                if source_rank is None or target_rank > source_rank:
                    violations.append(f"{path.relative_to(ENGINE.parent)}:{node.lineno} imports {target}")

    assert not violations, (
        "Upward imports (ADR 0002: pipeline > services > controllers > repositories > models; "
        "config/notify import no layer):\n  " + "\n  ".join(violations)
    )


def test_every_module_sits_in_a_layer_or_is_shared():
    stray = []
    for path in sorted(ENGINE.iterdir()):
        name = path.stem if path.suffix == ".py" else path.name
        if name.startswith(("__", ".")) or (path.is_file() and path.suffix != ".py"):
            continue
        if name not in RANK and name not in SHARED:
            stray.append(str(path.relative_to(ENGINE.parent)))

    assert not stray, (
        "Not a layer and not a shared module (ADR 0002 §3). Put the code in its layer, "
        "or add a new version of ADR 0002 that introduces this name:\n  " + "\n  ".join(stray)
    )


def test_no_folder_is_named_for_a_sequence_step():
    named = [
        str(path.relative_to(ENGINE.parent))
        for path in ENGINE.rglob("*")
        if path.is_dir() and path.name in SEQUENCE_NAMES
    ]

    assert not named, (
        "Folders named for when, not what (ADR 0002 §7):\n  " + "\n  ".join(named)
    )
