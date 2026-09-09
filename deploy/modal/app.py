"""Modal deployment and local batch entrypoint for the MuVisual workflow."""

from __future__ import annotations

import os
import sys
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import quote

import modal
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from deploy.modal.archive import SUPPORTED_EXTENSIONS, extract_result, zip_directory

APP_NAME = "muvisual-workflow"
CONFIG_PATH = Path(
    os.environ.get("MUVISUAL_MODAL_CONFIG", Path(__file__).with_name("config.toml"))
)
with CONFIG_PATH.open("rb") as config_file:
    GPU_TYPE = tomllib.load(config_file)["compute"]["gpu"]
if not isinstance(GPU_TYPE, str) or not GPU_TYPE.strip():
    raise ValueError("Modal config compute.gpu must be a non-empty string")
REMOTE_CONFIG_PATH = "/root/modal-config.toml"
CACHE_DIR = "/cache"
PROJECT_DIR = "/root/muvisual"
MODEL_DIR = f"{PROJECT_DIR}/data/model"

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("muvisual-model-cache", create_if_missing=True)
read_only_model_cache = model_cache.with_mount_options(read_only=True)
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi", "python-multipart")
    .add_local_file(CONFIG_PATH, remote_path=REMOTE_CONFIG_PATH, copy=True)
    .env({"MUVISUAL_MODAL_CONFIG": REMOTE_CONFIG_PATH})
    .add_local_python_source("deploy.modal.archive", copy=True)
)

# uv_sync uploads only the dependency manifests. Add the runtime config and
# Python package separately so local data, caches, and virtualenvs stay local.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .uv_sync()
    .add_local_file(CONFIG_PATH, remote_path=REMOTE_CONFIG_PATH, copy=True)
    .add_local_dir("config", remote_path=f"{PROJECT_DIR}/config", copy=True)
    .add_local_python_source("muvisual_workflow", copy=True)
    .add_local_python_source("deploy.modal.archive", copy=True)
    .run_commands(
        f"mkdir -p {PROJECT_DIR}/data {PROJECT_DIR}/temp",
        f"ln -s {CACHE_DIR} {MODEL_DIR}",
    )
    .env(
        {
            "MUVISUAL_MODAL_CONFIG": REMOTE_CONFIG_PATH,
            "MUVISUAL_PROJECT_ROOT": PROJECT_DIR,
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "HF_HUB_CACHE": f"{CACHE_DIR}/huggingface/hub",
            "TORCH_HOME": f"{CACHE_DIR}/torch",
            "AUDIO_SEPARATOR_MODEL_DIR": f"{CACHE_DIR}/BS-Roformer-SW",
            "TZ": "Asia/Shanghai",
            "TMPDIR": f"{PROJECT_DIR}/temp",
        }
    )
)


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={CACHE_DIR: model_cache},
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
    ],
)
def warmup_models() -> None:
    """Download all runtime model weights into the shared Volume once."""
    from dataclasses import replace

    from muvisual_workflow.audio_to_midi import AudioToMidiStep
    from muvisual_workflow.beat_detection import BeatDetector
    from muvisual_workflow.separation import prepare_local_model
    from muvisual_workflow.core.logging import configure_logging, get_logger
    from muvisual_workflow.music_metadata.chord_recognition.chord_cnn_lstm import (
        resolve_repository,
    )
    from muvisual_workflow.workflow.pipeline import TEMP_DIR, load_workflow_configs

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging()
    logger = get_logger("modal.warmup")
    warmed: set[tuple[str, ...]] = set()
    for config in load_workflow_configs():
        if not config.enabled:
            continue
        logger.info("Preparing models for workflow: %s", config.instrument)
        separation = config.separation
        if separation is not None and separation.enabled:
            key = ("separation", separation.model)
            if key not in warmed:
                prepare_local_model()
                warmed.add(key)

        beat_config = config.beat_detection
        if beat_config is not None and beat_config.enabled:
            key = (beat_config.algorithm, beat_config.model)
            if key not in warmed:
                # Madmom ships its weights with the installed package.
                if beat_config.algorithm == "beat_this":
                    detector = BeatDetector(
                        beat_config.model, device="cpu", dbn=beat_config.dbn
                    )
                    detector.release()
                warmed.add(key)

        audio_to_midi = config.audio_to_midi
        if audio_to_midi is not None and audio_to_midi.enabled:
            for instrument_config in audio_to_midi.instruments.values():
                key = (
                    "audio_to_midi", instrument_config.model, instrument_config.checkpoint
                )
                if key in warmed:
                    continue
                cpu_config = replace(
                    instrument_config,
                    device="cpu",
                    dtype="float32" if instrument_config.model == "muscriptor" else None,
                )
                step = AudioToMidiStep(cpu_config)
                step.release()
                warmed.add(key)

        for metadata_config in config.music_metadata:
            chord_config = metadata_config.chord_recognition
            if metadata_config.enabled and chord_config is not None:
                repository = resolve_repository(chord_config)
                warmed.add(
                    ("chord_cnn_lstm", str(repository), chord_config.chord_dictionary)
                )

    model_cache.commit()
    logger.info(
        "Model warmup complete: %s",
        ", ".join(":".join(key) for key in sorted(warmed)),
    )


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=20 * 60,
    volumes={CACHE_DIR: read_only_model_cache},
    secrets=[
        modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])
    ],
)
def process_audio_file(payload: bytes, suffix: str) -> tuple[str, bytes]:
    """Process one serialized audio upload in a GPU container."""
    from muvisual_workflow.core.logging import configure_logging
    from muvisual_workflow.workflow.pipeline import (
        TEMP_DIR,
        load_workflow_configs,
        process_audio_workflows,
        read_output_name,
    )

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio extension: {suffix}")
    if not payload:
        raise ValueError("Audio file is empty")

    # Reused containers must see the latest committed warmup before loading models.
    read_only_model_cache.reload()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging()
    configs = load_workflow_configs()
    if not any(config.enabled for config in configs):
        raise RuntimeError("No enabled workflows found")
    with tempfile.TemporaryDirectory(prefix="muvisual-api-", dir=TEMP_DIR) as temp_dir:
        work_root = Path(temp_dir)
        source = work_root / f"input{suffix}"
        source.write_bytes(payload)
        output_name = read_output_name(source)
        output_root = work_root / "output"
        output_root.mkdir()
        process_audio_workflows(
            source,
            output_name,
            output_root,
            work_root / "work",
            configs,
        )
        archive = zip_directory(output_root / output_name)
    return output_name, archive


@app.function(image=web_image)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(requires_proxy_auth=True)
def api() -> FastAPI:
    """Expose submission and polling routes through one Modal web endpoint."""
    web_app = FastAPI()

    @web_app.post("/submit")
    async def submit(file: UploadFile = File(...)) -> dict[str, str]:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise HTTPException(415, f"Unsupported audio extension; use: {supported}")

        payload = await file.read()
        if not payload:
            raise HTTPException(400, "Uploaded audio file is empty")

        call = await process_audio_file.spawn.aio(payload, suffix)
        return {"call_id": call.object_id}

    @web_app.get("/result/{call_id}")
    async def result(call_id: str) -> Response:
        function_call = modal.FunctionCall.from_id(call_id)
        try:
            output_name, archive = await function_call.get.aio(timeout=0)
        except modal.exception.OutputExpiredError as exc:
            raise HTTPException(404, "Processing result expired") from exc
        except modal.exception.NotFoundError as exc:
            raise HTTPException(404, "Processing result not found") from exc
        except TimeoutError:
            return Response(status_code=202)
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

    return web_app


@app.local_entrypoint()
def main(
    input_dir: str = "data/input",
    output_dir: str = "data/output",
    warmup_only: bool = False,
) -> None:
    """Process local input files through one Modal request per audio file."""
    if warmup_only:
        warmup_models.remote()
        print("Model warmup completed")
        return

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

    warmup_models.remote()
    print("Model warmup completed")

    failures: list[tuple[Path, str]] = []
    for index, audio_path in enumerate(audio_files, start=1):
        print(f"[{index}/{len(audio_files)}] Uploading: {audio_path}")
        try:
            output_name, archive = process_audio_file.remote(
                audio_path.read_bytes(), audio_path.suffix.lower()
            )
            result_path = extract_result(archive, destination_dir, output_name)
        except Exception as exc:
            failures.append((audio_path, str(exc)))
            print(f"Failed: {audio_path}: {exc}", file=sys.stderr)
        else:
            print(f"Completed: {result_path}")

    if failures:
        details = "\n".join(f"  {path}: {error}" for path, error in failures)
        raise RuntimeError(f"{len(failures)} file(s) failed:\n{details}")
