"""Execute the tutorial using this Python interpreter; save outputs outside code."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from paper_paths import new_output


def main():
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout per code cell in seconds")
    parser.add_argument("--notebook", default="01_raw_seeg_to_fig1.ipynb")
    args = parser.parse_args()
    target = new_output("executed_notebooks") / (Path(args.notebook).stem + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + ".ipynb")
    notebook = nbformat.read(CODE / "notebooks" / args.notebook, as_version=4)
    with tempfile.TemporaryDirectory(prefix="paper-notebook-kernel-") as temp:
        spec = Path(temp) / "paper-code"; spec.mkdir()
        (spec / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "Paper code", "language": "python"}))
        manager = KernelManager(kernel_name="paper-code", kernel_spec_manager=KernelSpecManager(kernel_dirs=[temp]))
        client = NotebookClient(notebook, km=manager, timeout=args.timeout,
                                resources={"metadata": {"path": str(CODE / "notebooks")}})
        try:
            client.execute()
        finally:
            nbformat.write(notebook, target)
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    print(json.dumps({"status": "PASS", "executed_notebook": str(target)}, indent=2))


if __name__ == "__main__":
    main()
