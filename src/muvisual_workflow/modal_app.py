"""Modal API and local batch client for the MuVisual workflow."""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import modal
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import Response


APP_NAME = "muvisual-workflow"
GPU_TYPE = "L40S"
CACHE_DIR = "/cache"
PROJECT_DIR = "/root/muvisual"
SUPPORTED_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".opus",
    ".m4a",
    ".aiff",
    ".ac3",
}

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("muvisual-model-cache", create_if_missing=True)
web_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "fastapi", "python-multipart"
)

# uv_sync uploads only the dependency manifests. Add the runtime config and
# Python package separately so local data, caches, and virtualenvs stay local.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .uv_sync()
    .add_local_dir("config", remote_path=f"{PROJECT_DIR}/config", copy=True)
    .add_local_python_source("muvisual_workflow", copy=True)
    .env(
        {
            "MUVISUAL_PROJECT_ROOT": PROJECT_DIR,
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "TORCH_HOME": f"{CACHE_DIR}/torch",
            "AUDIO_SEPARATOR_MODEL_DIR": f"{CACHE_DIR}/BS-Roformer-SW",
        }
    )
)


def _zip_directory(directory: Path) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(directory.parent))
    return archive.getvalue()


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60,
    volumes={CACHE_DIR: model_cache},
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
    ],
)
def process_audio_file(payload: bytes, suffix: str) -> tuple[str, bytes]:
    """Process one serialized audio upload in a GPU container."""
    from muvisual_workflow.audio.separation import prepare_local_model
    from muvisual_workflow.core.config import load_config
    from muvisual_workflow.workflow.pipeline import (
        TEMP_DIR,
        process_audio as run_pipeline,
        read_output_name,
        resolve_instrument_configs,
    )

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio extension: {suffix}")
    if not payload:
        raise ValueError("Audio file is empty")

    prepare_local_model()
    model_cache.commit()

    config = load_config()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="muvisual-api-", dir=TEMP_DIR) as temp_dir:
        work_root = Path(temp_dir)
        source = work_root / f"input{suffix}"
        source.write_bytes(payload)
        output_name = read_output_name(source)
        output_root = work_root / "output"
        output_root.mkdir()
        run_pipeline(
            source,
            output_name,
            output_root,
            work_root / "work",
            config.separation_model,
            resolve_instrument_configs(config.audio_to_midi, _ApiArgs()),
            config.beat_detection,
        )
        model_cache.commit()
        archive = _zip_directory(output_root / output_name)
    return output_name, archive


@app.function(image=web_image, timeout=60 * 60)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
async def process_audio(file: UploadFile = File(...)) -> Response:
    """Accept one tagged audio file and return its output directory as ZIP."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(415, f"Unsupported audio extension; use: {supported}")

    payload = await file.read()
    if not payload:
        raise HTTPException(400, "Uploaded audio file is empty")

    try:
        output_name, archive = await process_audio_file.remote.aio(payload, suffix)
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc

    return Response(
        content=archive,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                'attachment; filename="muvisual-output.zip"; '
                f"filename*=UTF-8''{quote(f'{output_name}.zip')}"
            )
        },
    )


class _ApiArgs:
    """Pipeline override namespace representing API defaults."""

    model = None
    audio_to_midi_model = None
    audio_to_midi_checkpoint = None
    device = None
    segment_hop_size = None
    segment_size = None


def _extract_result(archive: bytes, output_dir: Path, output_name: str) -> Path:
    """Validate and atomically replace one extracted result directory."""
    destination = output_dir / output_name
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        members = zipped.infolist()
        if not members:
            raise RuntimeError("Modal returned an empty ZIP archive")
        for member in members:
            parts = PurePosixPath(member.filename).parts
            if not parts or parts[0] != output_name or ".." in parts:
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


@app.local_entrypoint()
def main(input_dir: str = "data/input", output_dir: str = "data/output") -> None:
    """Process local input files through one Modal request per audio file."""
    source_dir = Path(input_dir).expanduser().resolve()
    destination_dir = Path(output_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {source_dir}")

    audio_files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found in: {source_dir}")

    failures: list[tuple[Path, str]] = []
    for index, audio_path in enumerate(audio_files, start=1):
        print(f"[{index}/{len(audio_files)}] Uploading: {audio_path}")
        try:
            output_name, archive = process_audio_file.remote(
                audio_path.read_bytes(), audio_path.suffix.lower()
            )
            result_path = _extract_result(archive, destination_dir, output_name)
        except Exception as exc:
            failures.append((audio_path, str(exc)))
            print(f"Failed: {audio_path}: {exc}", file=sys.stderr)
        else:
            print(f"Completed: {result_path}")

    if failures:
        details = "\n".join(f"  {path}: {error}" for path, error in failures)
        raise RuntimeError(f"{len(failures)} file(s) failed:\n{details}")
