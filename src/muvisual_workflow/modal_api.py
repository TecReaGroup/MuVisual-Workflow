"""Upload local audio files to the deployed Modal API."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
import uuid
import zipfile

from dotenv import dotenv_values

from muvisual_workflow.core.paths import PROJECT_ROOT
from muvisual_workflow.modal_app import SUPPORTED_EXTENSIONS, _extract_result


ENV_PATH = PROJECT_ROOT / ".env"


def _load_endpoint_url() -> str:
    endpoint_url = _load_env_value("MODAL_URL")
    if not endpoint_url or not isinstance(endpoint_url, str):
        raise RuntimeError(f"MODAL_URL is not configured in the environment or {ENV_PATH}")

    endpoint_url = endpoint_url.strip().rstrip("/")
    parsed = urlsplit(endpoint_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("MODAL_URL must be an absolute HTTP(S) URL")
    return endpoint_url


def _load_env_value(name: str) -> str | None:
    values = dotenv_values(ENV_PATH)
    return os.environ.get(name) or os.environ.get(name.replace("_", "-")) or values.get(
        name
    ) or values.get(name.replace("_", "-"))


def _load_proxy_auth() -> tuple[str, str]:
    key = _load_env_value("MODAL_KEY")
    secret = _load_env_value("MODAL_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "MODAL_KEY and MODAL_SECRET must be configured in the environment or "
            f"{ENV_PATH}"
        )
    return key.strip(), secret.strip()


def _post_audio(
    base_url: str,
    audio_path: Path,
    modal_key: str,
    modal_secret: str,
) -> str:
    boundary = uuid.uuid4().hex
    filename = f"input{audio_path.suffix.lower()}"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode("ascii")
    body = prefix + audio_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("ascii")
    request = Request(
        f"{base_url}/submit",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Modal-Key": modal_key,
            "Modal-Secret": modal_secret,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Modal submit API returned HTTP {exc.code}: {detail}") from exc

    call_id = result.get("call_id") if isinstance(result, dict) else None
    if not isinstance(call_id, str) or not call_id:
        raise RuntimeError("Modal submit API returned an invalid call_id")
    return call_id


def _poll_result(
    base_url: str,
    call_id: str,
    modal_key: str,
    modal_secret: str,
) -> bytes:
    request_url = f"{base_url}/result/{quote(call_id, safe='')}"
    headers = {"Modal-Key": modal_key, "Modal-Secret": modal_secret}

    while True:
        request = Request(request_url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=60) as response:
                if response.status == 202:
                    time.sleep(2)
                    continue
                archive = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Modal result API returned HTTP {exc.code}: {detail}"
            ) from exc

        content_type = response.headers.get_content_type()
        if content_type != "application/zip":
            raise RuntimeError(
                f"Modal result API returned unexpected content type: {content_type}"
            )
        return archive


def _read_output_name(archive: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        roots = {
            parts[0]
            for member in zipped.infolist()
            if (parts := PurePosixPath(member.filename).parts)
        }
    if len(roots) != 1:
        raise RuntimeError("Modal ZIP must contain exactly one top-level directory")
    return roots.pop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the deployed MuVisual Modal API.")
    parser.add_argument("--input", type=Path, default=Path("data/input"))
    parser.add_argument("--output", type=Path, default=Path("data/output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    audio_files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found in: {input_dir}")

    endpoint_url = _load_endpoint_url()
    modal_key, modal_secret = _load_proxy_auth()

    failures: list[tuple[Path, str]] = []
    for index, audio_path in enumerate(audio_files, start=1):
        print(f"[{index}/{len(audio_files)}] Uploading: {audio_path}")
        try:
            call_id = _post_audio(endpoint_url, audio_path, modal_key, modal_secret)
            archive = _poll_result(endpoint_url, call_id, modal_key, modal_secret)
            output_name = _read_output_name(archive)
            result_path = _extract_result(archive, output_dir, output_name)
        except Exception as exc:
            failures.append((audio_path, str(exc)))
            print(f"Failed: {audio_path}: {exc}", file=sys.stderr)
        else:
            print(f"Completed: {result_path}")

    if failures:
        details = "\n".join(f"  {path}: {error}" for path, error in failures)
        raise RuntimeError(f"{len(failures)} file(s) failed:\n{details}")


if __name__ == "__main__":
    main()
