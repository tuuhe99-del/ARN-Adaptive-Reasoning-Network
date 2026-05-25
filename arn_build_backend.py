"""Small PEP 517/660 backend for ARN's local-first editable installs."""

from __future__ import annotations

import base64
import csv
import hashlib
import os
from pathlib import Path
import shutil
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parent


def _project():
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]


def _dist_name() -> str:
    return _project()["name"].replace("-", "_")


def _version() -> str:
    return _project()["version"]


def _dist_info() -> str:
    return f"{_dist_name()}-{_version()}.dist-info"


def _wheel_name() -> str:
    return f"{_dist_name()}-{_version()}-py3-none-any.whl"


def _metadata_text() -> str:
    project = _project()
    lines = [
        "Metadata-Version: 2.3",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project.get('description', '')}",
        f"Requires-Python: {project.get('requires-python', '')}",
    ]
    for author in project.get("authors", []):
        if author.get("name"):
            lines.append(f"Author: {author['name']}")
    for classifier in project.get("classifiers", []):
        lines.append(f"Classifier: {classifier}")
    for dependency in project.get("dependencies", []):
        lines.append(f"Requires-Dist: {dependency}")
    lines.append("")
    return "\n".join(lines)


def _wheel_text() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: arn-build-backend",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )


def _entry_points_text() -> str:
    scripts = _project().get("scripts", {})
    lines = ["[console_scripts]"]
    for name, target in scripts.items():
        lines.append(f"{name} = {target}")
    lines.append("")
    return "\n".join(lines)


def _write_dist_info(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "METADATA").write_text(_metadata_text(), encoding="utf-8")
    (path / "WHEEL").write_text(_wheel_text(), encoding="utf-8")
    (path / "entry_points.txt").write_text(_entry_points_text(), encoding="utf-8")


def _hash_record(data: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}", str(len(data))


def _write_wheel(wheel_path: Path, files: dict[str, bytes]) -> None:
    record_name = f"{_dist_info()}/RECORD"
    rows = []
    for name, data in files.items():
        digest, size = _hash_record(data)
        rows.append([name, digest, size])
    rows.append([record_name, "", ""])

    record_lines = []
    for row in rows:
        record_lines.append(",".join(_csv_cell(cell) for cell in row))
    files[record_name] = ("\n".join(record_lines) + "\n").encode("utf-8")

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def _csv_cell(value: str) -> str:
    out = []
    csv.writer(out := _CsvSink()).writerow([value])
    return out.value.rstrip("\r\n")


class _CsvSink:
    def __init__(self) -> None:
        self.value = ""

    def write(self, value: str) -> None:
        self.value += value


def _dist_info_files() -> dict[str, bytes]:
    base = _dist_info()
    return {
        f"{base}/METADATA": _metadata_text().encode("utf-8"),
        f"{base}/WHEEL": _wheel_text().encode("utf-8"),
        f"{base}/entry_points.txt": _entry_points_text().encode("utf-8"),
    }


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    dist_info = _dist_info()
    target = Path(metadata_directory) / dist_info
    _write_dist_info(target)
    (target / "RECORD").write_text("", encoding="utf-8")
    return dist_info


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    files = _dist_info_files()
    files[f"{_dist_name()}-editable.pth"] = (str(ROOT) + os.linesep).encode("utf-8")
    wheel_path = Path(wheel_directory) / _wheel_name()
    _write_wheel(wheel_path, files)
    return wheel_path.name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    files = _dist_info_files()
    package_root = ROOT / "arn_v9"
    for path in package_root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            files[str(path.relative_to(ROOT))] = path.read_bytes()
    wheel_path = Path(wheel_directory) / _wheel_name()
    _write_wheel(wheel_path, files)
    return wheel_path.name


def build_sdist(sdist_directory, config_settings=None):
    shutil.make_archive(str(Path(sdist_directory) / f"{_dist_name()}-{_version()}"), "gztar", ROOT)
    return f"{_dist_name()}-{_version()}.tar.gz"
