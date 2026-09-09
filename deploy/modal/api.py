"""Upload local audio files to the deployed Modal API."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from http.client import HTTPException
import io
import json
import os
from pathlib import Path, PurePosixPath
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener, getproxies, proxy_bypass
import uuid
import zipfile

from dotenv import dotenv_values

from muvisual_workflow.core.paths import PROJECT_ROOT
from muvisual_workflow.core.logging import configure_logging, get_logger
from deploy.modal.archive import SUPPORTED_EXTENSIONS, extract_result


ENV_PATH = PROJECT_ROOT / ".env"
REQUEST_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 2
LOGGER = get_logger("modal.api")


class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Prevent credential forwarding and implicit POST redirects."""
        return None


@contextmanager
def _open_request(request: Request):
    """Report transport failures without exposing credentials or response bodies."""
    started = time.monotonic()
    operation = f"{request.get_method()} {urlsplit(request.full_url).path}"
    LOGGER.debug("Request started: %s", operation)
    try:
        with build_opener(_RejectRedirect()).open(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            LOGGER.debug("Response headers: %s HTTP=%s", operation, response.status)
            yield response
    except HTTPError as exc:
        exc.close()
        hint = {
            401: "Check Modal Proxy Token credentials (not SDK tokens).",
            403: "Check Proxy Token workspace/environment permissions.",
            404: "Check MODAL_URL against the deployed ASGI api endpoint URL.",
            413: "The gateway rejected the upload size.",
        }.get(exc.code, "Check Modal web endpoint logs.")
        if 300 <= exc.code < 400:
            hint = "Redirect refused; configure the final ASGI endpoint URL."
        raise RuntimeError(f"{operation}: HTTP {exc.code}. {hint}") from None
    except (URLError, OSError, HTTPException) as exc:
        reason = exc.reason if isinstance(exc, URLError) else exc
        if isinstance(reason, ssl.SSLCertVerificationError):
            hint = "TLS certificate verification failed; check the trust store."
        elif isinstance(reason, ssl.SSLError):
            hint = "TLS negotiation failed; inspect the network path."
        elif isinstance(reason, TimeoutError):
            hint = f"Socket operation exceeded {REQUEST_TIMEOUT_SECONDS}s."
        else:
            hint = "Connection failed; the peer, proxy or network may have interrupted it."
        raise RuntimeError(
            f"{operation}: {type(reason).__name__} "
            f"errno={getattr(reason, 'errno', None)} "
            f"winerror={getattr(reason, 'winerror', None)}. {hint} "
            "TLS verification and system proxy settings were not bypassed."
        ) from None
    finally:
        LOGGER.debug("Request ended: %s elapsed=%.2fs", operation, time.monotonic() - started)


def _load_endpoint_url() -> str:
    endpoint_url = _load_env_value("MODAL_URL")
    if not endpoint_url or not isinstance(endpoint_url, str):
        raise RuntimeError(f"MODAL_URL is not configured in the environment or {ENV_PATH}")

    endpoint_url = endpoint_url.strip().rstrip("/")
    try:
        parsed = urlsplit(endpoint_url)
        valid = (
            parsed.scheme == "https" and parsed.hostname and parsed.port != 0
            and parsed.username is None and parsed.password is None
            and not parsed.query and not parsed.fragment
            and not any(character.isspace() for character in endpoint_url)
        )
    except ValueError:
        valid = False
    if not valid:
        raise RuntimeError(
            "MODAL_URL must be the absolute HTTPS base URL of the deployed ASGI api, "
            "without credentials, query or fragment"
        )
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
    LOGGER.info("Submitting audio: multipart_bytes=%s", len(body))
    with _open_request(request) as response:
        result = json.loads(response.read())

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
        with _open_request(request) as response:
            if response.status == 202:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            LOGGER.info("Downloading result: call_id=%s", call_id)
            archive = response.read()

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
    configure_logging()
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
    endpoint = urlsplit(endpoint_url)
    LOGGER.info(
        "Endpoint host=%s scheme=%s proxy_configured=%s proxy_bypass=%s "
        "socket_timeout=%ss poll_interval=%ss; polling has no overall deadline",
        endpoint.hostname, endpoint.scheme, endpoint.scheme in getproxies(),
        proxy_bypass(endpoint.netloc), REQUEST_TIMEOUT_SECONDS, POLL_INTERVAL_SECONDS,
    )

    failures: list[tuple[Path, str]] = []
    for index, audio_path in enumerate(audio_files, start=1):
        LOGGER.info("[%s/%s] Preparing upload: %s", index, len(audio_files), audio_path)
        stage = "submission"
        call_id = None
        started = time.monotonic()
        try:
            call_id = _post_audio(endpoint_url, audio_path, modal_key, modal_secret)
            stage = "polling/download"
            LOGGER.info("Submission accepted: call_id=%s; polling result", call_id)
            archive = _poll_result(endpoint_url, call_id, modal_key, modal_secret)
            stage = "extraction"
            output_name = _read_output_name(archive)
            result_path = extract_result(archive, output_dir, output_name)
        except Exception as exc:
            failures.append((audio_path, str(exc)))
            LOGGER.error("Failed: %s stage=%s elapsed=%.2fs: %s", audio_path, stage, time.monotonic() - started, exc)
            if stage == "submission":
                LOGGER.warning("Submission outcome may be unknown; no retry was made. Check Modal calls before submitting this file again.")
            elif stage == "polling/download":
                LOGGER.warning("Remote call may still be running: call_id=%s. Do not resubmit the audio to recover a download.", call_id)
        else:
            LOGGER.info("Completed: %s elapsed=%.2fs", result_path, time.monotonic() - started)

    if failures:
        details = "\n".join(f"  {path}: {error}" for path, error in failures)
        raise RuntimeError(f"{len(failures)} file(s) failed:\n{details}")


if __name__ == "__main__":
    main()
