"""Pack workflow output and validate downloaded archives without Modal imports."""

from __future__ import annotations

import io
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".ac3"}


def zip_directory(directory: Path) -> bytes:
    """Archive a result directory including its top-level name."""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(directory.parent))
    return archive.getvalue()


def extract_result(archive: bytes, output_dir: Path, output_name: str) -> Path:
    """Validate and stage an archive before replacing its output directory."""
    if (
        not output_name
        or output_name in {".", ".."}
        or any(character in output_name for character in '/\\:\x00')
    ):
        raise RuntimeError(f"Unsafe ZIP output name: {output_name!r}")
    destination = output_dir / output_name
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        members = zipped.infolist()
        if not members:
            raise RuntimeError("Modal returned an empty ZIP archive")
        for member in members:
            member_path = PurePosixPath(member.filename)
            parts = member_path.parts
            if (
                not parts
                or member_path.is_absolute()
                or parts[0] != output_name
                or ".." in parts
                or "\\" in member.filename
                or ":" in member.filename
            ):
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")

        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="muvisual-download-", dir=output_dir
        ) as temp_dir:
            staging_root = Path(temp_dir)
            zipped.extractall(staging_root)
            staged_result = staging_root / output_name
            if not staged_result.is_dir():
                raise RuntimeError(
                    f"ZIP does not contain the expected directory: {output_name}"
                )
            if destination.exists():
                if not destination.is_dir():
                    raise RuntimeError(f"Output path is not a directory: {destination}")
                shutil.rmtree(destination)
            shutil.move(str(staged_result), destination)
    return destination
