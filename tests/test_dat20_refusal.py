"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import importlib
import io
import json
import os
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/dinov2_feature_extraction_colab.ipynb"
BYOD_CELL = "dinov2_feature_extraction-17"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


SOURCE = _cell(BYOD_CELL)
STEM = "dinov2_feature_extraction"
_PACKAGE = next(p.name for p in (ROOT / "src").iterdir() if (p / "samples.py").is_file())
MODULES = [importlib.import_module(f"{_PACKAGE}.{p.stem}") for p in sorted((ROOT / "src" / _PACKAGE).glob("*.py")) if p.stem != "__init__"]
EARLIER = {"splits": "old", "train_records": "old", "test_records": "old", "data_source": "old", "adapted_test": "old", "dataset_manifests": "old"}


def _png(value: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (value, (value * 7) % 256, (value * 13) % 256)).save(buffer, format="PNG")
    return buffer.getvalue()


def _zip(entries: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _dataset(n: int, *, missing: int | None = None, broken: int | None = None, columns: str = "id,file,label") -> dict:
    rows, entries = [columns], {}
    for i in range(n):
        rows.append(f"r{i},img{i}.png,{'cat' if i % 2 else 'dog'}")
        if i != missing:
            entries[f"img{i}.png"] = b"text" if i == broken else _png(10 + 11 * i)
    return {"labels.csv": "\n".join(rows) + "\n", **entries}


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    (out / f"{STEM}_adapter").mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "input_manifest.json", "evaluation_report.json", "result.json", "predictions.csv"):
        (out / f"{STEM}_{name}").write_text("{}", encoding="utf-8")


def _namespace() -> dict:
    ns: dict = {"os": os, "Path": Path, "Image": Image}
    for module in MODULES:
        ns.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    ns.update(EARLIER, pipe="kept")
    return ns


def _run(monkeypatch, tmp_path, path: str, *, use_byod=True, colab=None, ns=None, source=None) -> dict:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    _seed_outputs(tmp_path)
    source = (source or SOURCE).replace("USE_BYOD = False", f"USE_BYOD = {use_byod}", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {path!r}", 1)
    ns = ns if ns is not None else _namespace()
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _colab(uploaded: dict):
    module = types.ModuleType("google.colab")
    module.files = types.SimpleNamespace(upload=lambda: uploaded)
    return module


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime-host\n", encoding="utf-8")
    (tmp_path / "sample_data").mkdir()
    (tmp_path / "sample_data" / "california_housing_test.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "fake.zip").write_bytes(b"not a zip")
    (tmp_path / "nolabels.zip").write_bytes(_zip({"img0.png": _png(1)}))
    (tmp_path / "columns.zip").write_bytes(_zip(_dataset(12, columns="id,path,label")))
    (tmp_path / "missing.zip").write_bytes(_zip(_dataset(12, missing=3)))
    (tmp_path / "broken.zip").write_bytes(_zip(_dataset(12, broken=2)))
    (tmp_path / "few.zip").write_bytes(_zip(_dataset(3)))
    return {
        "folder without labels.csv (the scripted /content/sample_data run)": ("sample_data", "the folder holds no labels.csv"),
        "text file": ("hostname", "must be a directory or a .zip"),
        "missing path": ("nope.zip", "does not exist in this runtime"),
        "corrupt zip": ("fake.zip", "BadZipFile"),
        "zip without labels.csv": ("nolabels.zip", "must contain labels.csv"),
        "missing columns": ("columns.zip", "missing columns"),
        "named image missing": ("missing.zip", "is not in the dataset"),
        "undecodable image": ("broken.zip", "ValueError"),
        "too few records": ("few.zip", "Error"),
    }


CASES = ["folder without labels.csv (the scripted /content/sample_data run)", "text file", "missing path", "corrupt zip", "zip without labels.csv", "missing columns", "named image missing", "undecodable image", "too few records"]


@pytest.mark.parametrize("case", CASES)
def test_byod_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    file_name, rule = _cases(tmp_path)[case]
    ns = _namespace()
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, str(tmp_path / file_name), ns=ns)
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert ns["pipe"] == "kept"


def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, "", colab=None)
    assert _refusal_line(caught.value).startswith("Refused: (no file): USE_BYOD opens the Colab upload dialog, and this runtime is not Colab")


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [({}, "(no file)", "(got 0)"), ({"a.zip": b"1", "b.zip": b"2"}, "a.zip, b.zip", "(got 2)")])
def test_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, "", colab=_colab(uploaded))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: Upload exactly one .zip holding labels.csv") and rule in line


def test_uploaded_file_is_named_in_the_refusal(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, "", colab=_colab({"mine.zip": b"not a zip"}))
    assert _refusal_line(caught.value).startswith("Refused: mine.zip: ")


def test_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    (tmp_path / "ok.zip").write_bytes(_zip(_dataset(24)))
    ns = _namespace()

    def broken(path):
        raise RuntimeError("internal bug")

    ns["load_byod_dataset"] = broken
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _run(monkeypatch, tmp_path, str(tmp_path / "ok.zip"), ns=ns)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing is removed for an internal error


def test_sample_path_errors_are_not_refusals(monkeypatch, tmp_path):
    ns = _namespace()

    def broken(cache_dir):
        raise ValueError("pinned corpus digest mismatch")

    ns["fetch_corpus"] = broken
    with pytest.raises(ValueError, match="pinned corpus digest mismatch") as caught:
        _run(monkeypatch, tmp_path, "", use_byod=False, ns=ns)
    assert not getattr(caught.value, "dimer_refusal", False)


def test_good_byod_zip_reaches_the_split(monkeypatch, tmp_path):
    (tmp_path / "ok.zip").write_bytes(_zip(_dataset(24)))
    ns = _namespace()
    seen = {}

    def stop(records, seed):
        seen["n"] = len(records)
        raise RuntimeError("stop after loading")

    ns["split_dataset"] = stop
    with pytest.raises(RuntimeError, match="stop after loading"):
        _run(monkeypatch, tmp_path, str(tmp_path / "ok.zip"), ns=ns)
    assert seen["n"] == 24


@pytest.mark.skipif(not Path("/content/sample_data").is_dir(), reason="needs a fresh Colab runtime's /content/sample_data")
def test_scripted_refusal_input_is_the_real_sample_data_folder(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_PATH='/content/sample_data': a folder with no labels.csv."""
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, "/content/sample_data")
    assert _refusal_line(caught.value).startswith("Refused: sample_data: the folder holds no labels.csv")
