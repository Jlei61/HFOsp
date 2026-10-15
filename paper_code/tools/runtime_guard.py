"""Checks used to prove that the relocated notebook does not read the old repo."""
from pathlib import Path
import os
import sys


def install_guard():
    value = os.environ.get("HFO_PAPER_FORBID_ROOT")
    if not value:
        return
    forbidden = Path(value).resolve()

    def audit(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path == forbidden or forbidden in path.parents:
                raise RuntimeError(f"Relocation check blocked access to the original repository: {path}")
    sys.addaudithook(audit)


def local_import_report():
    from paper_paths import CODE_ROOT
    modules = {}
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] not in {"data_processing", "analysis", "modeling", "plotting", "paper_paths"}:
            continue
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if CODE_ROOT not in path.parents:
                raise AssertionError(f"Nonlocal code import: {name}: {path}")
            modules[name] = str(path.relative_to(CODE_ROOT))
    return {"all_project_imports_local": True, "modules_loaded": len(modules)}
