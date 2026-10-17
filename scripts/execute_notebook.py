"""Execute a benchmark notebook from a fresh kernel and retain its outputs."""

import argparse
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-timeout", type=float, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = args.notebook.resolve()
    root = Path(__file__).resolve().parents[1]
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ["BENCHMARK_PROFILE"] = args.profile
    os.environ["BENCHMARK_DEVICE"] = args.device
    os.environ["BENCHMARK_MODEL_TIMEOUT"] = str(args.model_timeout)
    for name, directory in {"JUPYTER_RUNTIME_DIR": "jupyter-runtime", "IPYTHONDIR": "ipython",
                            "MPLCONFIGDIR": "matplotlib"}.items():
        os.environ.setdefault(name, str(root / ".cache" / directory))
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(notebook, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    finally:
        output = args.output or path
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
    print(f"Executed {path.name}: {args.profile} profile; output={output}", flush=True)


if __name__ == "__main__":
    main()
