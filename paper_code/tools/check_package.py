"""Check copied source integrity, notebook cleanliness, and local imports."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for row in manifest["files"]:
        path = ROOT / row["destination"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["copied_sha256"]:
            raise ValueError(f"Copied source changed: {row['destination']}")
    python_files = list(ROOT.rglob("*.py"))
    for path in python_files:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and not node.level:
                name = node.module or ""
                if name == "src" or name == "scripts" or name.startswith(("src.", "scripts.")):
                    raise AssertionError(f"Original repository import remains: {path}: {name}")
        compile(tree, str(path), "exec")
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            raise AssertionError(f"Symlink is not an independent code copy: {path}")
        if path.suffix.lower() in {".npz", ".npy", ".csv", ".png", ".pdf", ".gif", ".tex", ".edf", ".data", ".head"}:
            raise AssertionError(f"Non-code artifact in code directory: {path}")
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        assert all(not c.get("outputs") and c.get("execution_count") is None
                   for c in notebook["cells"] if c["cell_type"] == "code")
    print(json.dumps({"status": "PASS", "copied_sources": len(manifest["files"]),
                      "python_files": len(python_files), "symlinks": 0, "notebook_outputs_cleared": True}, indent=2))


if __name__ == "__main__":
    main()
