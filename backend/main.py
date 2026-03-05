import csv
import io
import asyncio
import base64
import json
import re
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from llama_cpp import Llama

from pytocl.events import detect_events, enrich_telemetry_progress
from pytocl.laps import summarize_laps
from llm_commentary import run_pipeline_stream as run_llm_pipeline_stream
from audio_commentary import run_pipeline_stream as run_audio_pipeline_stream


"""
Main operations:
- GET /api/sessions (list sessions)
- POST /api/sessions (upload CSV, stores it, generates commentary)
"""

SESSIONS_DIR = Path("sessions")
SESSION_METADATA_FILENAME = "session_meta.json"
MODEL_PATH = Path("models", "granite_model.gguf")
COMMENTARY_BANK_PATH = Path("assets", "commentary_bank.csv")

origins = [
    "http://localhost:5173",  # dev environment
]

llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
job_queue: asyncio.Queue[str] = asyncio.Queue()
job_worker_task: asyncio.Task | None = None


class JobRecord(TypedDict):
    job_id: str
    job_type: Literal["text_commentary", "audio_commentary"]
    session_id: str
    session_dir: str
    status: str
    commentary_path: str
    pipeline_job_id: str
    audio_path: NotRequired[str]
    audio_job_id: NotRequired[str]
    progress: NotRequired[dict]
    progress_text: NotRequired[str]
    progress_stage: NotRequired[str]
    progress_status: NotRequired[str]
    progress_timestamp: NotRequired[str]
    error: NotRequired[str]


class PipelineJobRecord(TypedDict):
    pipeline_job_id: str
    session_id: str
    status: str
    text_job_id: str
    audio_job_id: str | None
    session_dir: str
    commentary_path: str
    audio_path: str
    error: NotRequired[str]


# tracks individual executable tasks (separate entries for text and audio generation)
jobs: dict[str, JobRecord] = {}
# tracks end-to-end session workflow (one entry per session)
pipeline_jobs: dict[str, PipelineJobRecord] = {}


class JobStatus(str, Enum):
    QUEUED = "queued"
    GENERATING_TEXT_COMMENTARY = "generating_text_commentary"
    GENERATING_AUDIO_COMMENTARY = "generating_audio_commentary"
    DONE = "done"
    FAILED = "failed"


class LLMReq(BaseModel):
    prompt: str
    max_tokens: int = Field(80, ge=1, le=512)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _sse_retry(milliseconds: int) -> str:
    return f"retry: {milliseconds}\n\n"


def _sse_comment(comment: str) -> str:
    return f": {comment}\n\n"


def _default_session_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"


def _session_id_from_name(name: str | None) -> str:
    if not name:
        return _default_session_id()

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        return _default_session_id()
    return cleaned


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_session_metadata(
    session_dir: Path, session_id: str, created_at: str
) -> None:
    metadata_path = session_dir / SESSION_METADATA_FILENAME
    metadata_path.write_text(
        json.dumps({"session_id": session_id, "created_at": created_at}),
        encoding="utf-8",
    )


def _read_session_created_at(session_dir: Path) -> str:
    metadata_path = session_dir / SESSION_METADATA_FILENAME
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            created_at = metadata.get("created_at")
            if isinstance(created_at, str) and created_at:
                return created_at
        except Exception:
            pass
    # Backward-compatible fallback for sessions created before metadata existed.
    return (
        datetime.fromtimestamp(session_dir.stat().st_mtime, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _iso_to_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _with_progress(payload: dict) -> dict:
    event = dict(payload)
    total = event.get("total_lines")
    processed = event.get("processed_lines")
    generated = event.get("generated_lines")

    if not isinstance(total, int) or total <= 0:
        return event

    completed: int | None = None
    if isinstance(processed, int):
        completed = processed
    elif isinstance(generated, int):
        completed = generated

    if completed is None:
        return event

    event["progress"] = {
        "completed": completed,
        "total": total,
        "percent": round(completed / total, 4),
    }
    if isinstance(processed, int):
        event["progress"]["processed_lines"] = processed
    if isinstance(generated, int):
        event["progress"]["generated_lines"] = generated
    event["progress_text"] = f"{completed}/{total}"
    return event


def _update_job_progress_from_event(job_id: str, event: dict) -> None:
    job = jobs.get(job_id)
    if not job:
        return

    normalized = _with_progress(event)
    progress = normalized.get("progress")
    if isinstance(progress, dict):
        job["progress"] = progress
        job["progress_text"] = normalized.get("progress_text", "")
        if "stage" in normalized:
            job["progress_stage"] = str(normalized["stage"])
        if "status" in normalized:
            job["progress_status"] = str(normalized["status"])
        if "timestamp" in normalized:
            job["progress_timestamp"] = str(normalized["timestamp"])


def _pipeline_snapshot_payload(pipeline: PipelineJobRecord) -> dict:
    text_job = jobs.get(pipeline["text_job_id"])
    audio_job = jobs.get(pipeline["audio_job_id"]) if pipeline["audio_job_id"] else None

    text_job_status = text_job["status"] if text_job else None
    audio_job_status = audio_job["status"] if audio_job else None
    if text_job_status is None and Path(pipeline["commentary_path"]).exists():
        text_job_status = JobStatus.DONE.value
    if audio_job_status is None and Path(pipeline["audio_path"]).exists():
        audio_job_status = JobStatus.DONE.value
    if text_job_status is None:
        text_job_status = JobStatus.QUEUED.value
    if audio_job_status is None:
        audio_job_status = JobStatus.QUEUED.value

    payload = {
        "stage": "pipeline",
        "status": pipeline["status"],
        "session_id": pipeline["session_id"],
        "pipeline_job_id": pipeline["pipeline_job_id"],
        "session_dir": pipeline["session_dir"],
        "commentary_path": pipeline["commentary_path"],
        "audio_path": pipeline["audio_path"],
        "text_job_id": pipeline["text_job_id"],
        "text_job_status": text_job_status,
        "audio_job_id": pipeline["audio_job_id"] or "",
        "audio_job_status": audio_job_status,
    }
    if "error" in pipeline:
        payload["error"] = pipeline["error"]
    if text_job and "error" in text_job:
        payload["text_error"] = text_job["error"]
    if audio_job and "error" in audio_job:
        payload["audio_error"] = audio_job["error"]

    progress_source: JobRecord | None = None
    progress_stage: str | None = None
    if (
        audio_job_status == JobStatus.GENERATING_AUDIO_COMMENTARY.value
        and audio_job
        and "progress" in audio_job
    ):
        progress_source = audio_job
        progress_stage = "audio_generation"
    elif (
        text_job_status == JobStatus.GENERATING_TEXT_COMMENTARY.value
        and text_job
        and "progress" in text_job
    ):
        progress_source = text_job
        progress_stage = "text_generation"
    elif audio_job and "progress" in audio_job:
        progress_source = audio_job
        progress_stage = "audio_generation"
    elif text_job and "progress" in text_job:
        progress_source = text_job
        progress_stage = "text_generation"

    if progress_source and isinstance(progress_source.get("progress"), dict):
        payload["progress"] = progress_source["progress"]
        payload["progress_stage"] = progress_stage or "pipeline"
        if "progress_text" in progress_source:
            payload["progress_text"] = progress_source["progress_text"]
    elif pipeline["status"] == JobStatus.DONE.value:
        payload["progress"] = {"completed": 1, "total": 1, "percent": 1.0}
        payload["progress_stage"] = "pipeline"
        payload["progress_text"] = "1/1"

    return payload


def _enqueue_audio_commentary_job(session_id: str, session_dir: Path) -> str:
    audio_job_id = uuid4().hex
    audio_out_path = session_dir / "commentary.wav"
    jobs[audio_job_id] = {
        "job_id": audio_job_id,
        "job_type": "audio_commentary",
        "session_id": session_id,
        "session_dir": str(session_dir),
        "status": JobStatus.QUEUED.value,
        "commentary_path": str(session_dir / "commentary.csv"),
        "audio_path": str(audio_out_path),
    }
    return audio_job_id


def _update_pipeline_status(
    pipeline_job_id: str, status: JobStatus, error: str | None = None
) -> None:
    pipeline = pipeline_jobs.get(pipeline_job_id)
    if not pipeline:
        return
    pipeline["status"] = status.value
    if error:
        pipeline["error"] = error
    elif "error" in pipeline:
        del pipeline["error"]


async def _run_text_commentary_job(job_id: str) -> None:
    job = jobs[job_id]
    job["status"] = JobStatus.GENERATING_TEXT_COMMENTARY.value
    pipeline_job_id = job["pipeline_job_id"]
    _update_pipeline_status(pipeline_job_id, JobStatus.GENERATING_TEXT_COMMENTARY)
    session_dir = Path(job["session_dir"])
    events_path = session_dir / "events.csv"
    commentary_path = session_dir / "commentary.csv"
    try:
        loop = asyncio.get_running_loop()

        def text_runner() -> None:
            for event in run_llm_pipeline_stream(
                llm=llm,
                csv_path=str(events_path),
                out_path=str(commentary_path),
                style_bank_path=str(COMMENTARY_BANK_PATH),
            ):
                loop.call_soon_threadsafe(
                    _update_job_progress_from_event, job_id, event
                )

        await asyncio.to_thread(text_runner)
        job["status"] = JobStatus.DONE.value
        job["commentary_path"] = str(commentary_path)
        audio_job_id = _enqueue_audio_commentary_job(
            session_id=job["session_id"], session_dir=session_dir
        )
        jobs[audio_job_id]["pipeline_job_id"] = pipeline_job_id
        job["audio_job_id"] = audio_job_id
        pipeline = pipeline_jobs.get(pipeline_job_id)
        if pipeline:
            pipeline["audio_job_id"] = audio_job_id
        _update_pipeline_status(pipeline_job_id, JobStatus.QUEUED)
        await job_queue.put(audio_job_id)
    except Exception as exc:
        job["status"] = JobStatus.FAILED.value
        job["error"] = str(exc)
        _update_pipeline_status(pipeline_job_id, JobStatus.FAILED, error=str(exc))


async def _run_audio_commentary_job(job_id: str) -> None:
    job = jobs[job_id]
    job["status"] = JobStatus.GENERATING_AUDIO_COMMENTARY.value
    pipeline_job_id = job["pipeline_job_id"]
    _update_pipeline_status(pipeline_job_id, JobStatus.GENERATING_AUDIO_COMMENTARY)
    commentary_path = job["commentary_path"]
    audio_path = job["audio_path"]
    try:
        loop = asyncio.get_running_loop()

        def audio_runner() -> None:
            for event in run_audio_pipeline_stream(
                commentary_csv=commentary_path, out_path=audio_path
            ):
                loop.call_soon_threadsafe(
                    _update_job_progress_from_event, job_id, event
                )

        await asyncio.to_thread(audio_runner)
        job["status"] = JobStatus.DONE.value
        _update_pipeline_status(pipeline_job_id, JobStatus.DONE)
    except Exception as exc:
        job["status"] = JobStatus.FAILED.value
        job["error"] = str(exc)
        print("FAAAAAHHHHHH")
        _update_pipeline_status(pipeline_job_id, JobStatus.FAILED, error=str(exc))


async def _job_worker() -> None:
    while True:
        job_id = await job_queue.get()
        try:
            job = jobs.get(job_id)
            if not job:
                continue
            if job.get("job_type") == "text_commentary":
                await _run_text_commentary_job(job_id)
            elif job.get("job_type") == "audio_commentary":
                await _run_audio_commentary_job(job_id)
            else:
                job["status"] = JobStatus.FAILED.value
                job["error"] = f"Unknown job type: {job.get('job_type')}"
                pipeline_job_id = job.get("pipeline_job_id")
                if pipeline_job_id:
                    _update_pipeline_status(
                        pipeline_job_id, JobStatus.FAILED, error=job["error"]
                    )
        finally:
            job_queue.task_done()


@app.on_event("startup")
async def startup_worker() -> None:
    global job_worker_task
    if job_worker_task is None or job_worker_task.done():
        job_worker_task = asyncio.create_task(_job_worker())


@app.on_event("shutdown")
async def shutdown_worker() -> None:
    global job_worker_task
    if job_worker_task is not None:
        job_worker_task.cancel()
        try:
            await job_worker_task
        except asyncio.CancelledError:
            pass
        job_worker_task = None


@app.get("/")
async def root():
    return {"message": "Salaam"}


@app.post("/api/sessions")
async def upload_csv(file: UploadFile = File(...), name: str | None = Form(None)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty")

    try:
        csv_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400, detail="CSV must be UTF-8 encoded"
        ) from exc

    # Each upload gets its own session folder on disk.
    safe_filename = Path("telemetry.csv").name
    session_id = _session_id_from_name(name)
    if (SESSIONS_DIR / session_id).exists():
        session_id = f"{session_id}_{uuid4().hex[:8]}"
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    created_at = _utc_now_iso()
    _write_session_metadata(
        session_dir=session_dir, session_id=session_id, created_at=created_at
    )
    file_path = session_dir / safe_filename
    file_path.write_bytes(raw_bytes)

    try:
        telemetry_df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not parse telemetry CSV: {exc}"
        ) from exc

    try:
        telemetry_df = enrich_telemetry_progress(telemetry_df)
        telemetry_df.to_csv(file_path, index=False)

        laps_df = summarize_laps(telemetry_df.copy())
        laps_path = session_dir / "laps.csv"
        laps_df.to_csv(laps_path, index=False)

        events_df = detect_events(telemetry_df.copy(), session_id=session_id)
        events_path = session_dir / "events.csv"
        events_df.to_csv(events_path, index=False)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not generate lap summaries/events from telemetry: {exc}",
        ) from exc

    telemetry_csv_text = telemetry_df.to_csv(index=False)
    reader = csv.DictReader(io.StringIO(telemetry_csv_text))
    rows = list(reader)

    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV must include a header row")

    # Queue commentary generation once events.csv is ready.
    pipeline_job_id = session_id
    job_id = uuid4().hex
    jobs[job_id] = {
        "job_id": job_id,
        "job_type": "text_commentary",
        "pipeline_job_id": pipeline_job_id,
        "session_id": session_id,
        "session_dir": str(session_dir),
        "status": JobStatus.QUEUED.value,
        "commentary_path": str(session_dir / "commentary.csv"),
    }
    pipeline_jobs[pipeline_job_id] = {
        "pipeline_job_id": pipeline_job_id,
        "session_id": session_id,
        "status": JobStatus.QUEUED.value,
        "text_job_id": job_id,
        "audio_job_id": None,
        "session_dir": str(session_dir),
        "commentary_path": str(session_dir / "commentary.csv"),
        "audio_path": str(session_dir / "commentary.wav"),
    }
    await job_queue.put(job_id)

    # Queue audio generation once commentary.csv is generated.

    return {
        "session_id": session_id,
        "created_at": created_at,
        "filename": file.filename,
        "saved_path": str(file_path),
        "laps_path": str(laps_path),
        "events_path": str(events_path),
        "commentary_path": str(session_dir / "commentary.csv"),
        "commentary_job_id": job_id,
        "commentary_job_status": JobStatus.QUEUED.value,
        "pipeline_job_id": pipeline_job_id,
        "pipeline_job_status": JobStatus.QUEUED.value,
        "audio_path": str(session_dir / "commentary.wav"),
        "columns": reader.fieldnames,
        "row_count": len(rows),
        "lap_count": len(laps_df),
        "event_count": len(events_df),
        "rows": rows,
    }


@app.get("/api/sessions/{session_id}/stream")
async def stream_session_pipeline(session_id: str, request: Request):
    pipeline = pipeline_jobs.get(session_id)
    if not pipeline:
        session_dir = SESSIONS_DIR / session_id
        if not session_dir.exists() or not session_dir.is_dir():
            raise HTTPException(status_code=404, detail="Session pipeline not found")

        commentary_path = session_dir / "commentary.csv"
        audio_path = session_dir / "commentary.wav"
        events_path = session_dir / "events.csv"

        if audio_path.exists():
            status = JobStatus.DONE.value
        elif commentary_path.exists():
            status = JobStatus.GENERATING_AUDIO_COMMENTARY.value
        elif events_path.exists():
            status = JobStatus.GENERATING_TEXT_COMMENTARY.value
        else:
            status = JobStatus.QUEUED.value

        pipeline = {
            "pipeline_job_id": session_id,
            "session_id": session_id,
            "status": status,
            "text_job_id": f"restored_text_{session_id}",
            "audio_job_id": f"restored_audio_{session_id}",
            "session_dir": str(session_dir),
            "commentary_path": str(commentary_path),
            "audio_path": str(audio_path),
        }
        pipeline_jobs[session_id] = pipeline

    def snapshot() -> dict:
        current = pipeline_jobs.get(session_id)
        if not current:
            return {
                "stage": "pipeline",
                "status": "failed",
                "session_id": session_id,
                "error": "Session pipeline disappeared",
            }
        return _pipeline_snapshot_payload(current)

    async def event_stream():
        # Ask clients to back off reconnect attempts if disconnected.
        yield _sse_retry(3000)
        yield _sse(
            {
                "stage": "session",
                "status": "ready",
                "session_id": pipeline["session_id"],
                "pipeline_job_id": pipeline["pipeline_job_id"],
                "session_dir": pipeline["session_dir"],
                "commentary_path": pipeline["commentary_path"],
                "audio_path": pipeline["audio_path"],
            }
        )

        last_payload: str | None = None
        last_heartbeat = asyncio.get_running_loop().time()

        while True:
            if await request.is_disconnected():
                break

            payload = snapshot()
            serialized = json.dumps(payload, sort_keys=True)
            if serialized != last_payload:
                yield _sse(payload)
                last_payload = serialized

            now = asyncio.get_running_loop().time()
            if now - last_heartbeat >= 10.0:
                yield _sse_comment("keep-alive")
                last_heartbeat = now

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sessions")
async def list_sessions():
    if not SESSIONS_DIR.exists():
        return {"sessions": []}

    sessions = [
        {"session_id": entry.name, "created_at": _read_session_created_at(entry)}
        for entry in SESSIONS_DIR.iterdir()
        if entry.is_dir()
    ]
    sessions.sort(key=lambda item: _iso_to_datetime(item["created_at"]), reverse=True)
    return {"sessions": sessions}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists() or not session_dir.is_dir():
        print(str(session_dir))
        raise HTTPException(status_code=404, detail="Session not found")

    pipeline = pipeline_jobs.pop(session_id, None)
    removed_job_ids: set[str] = set()
    if pipeline:
        removed_job_ids.add(pipeline["text_job_id"])
        if pipeline.get("audio_job_id"):
            removed_job_ids.add(str(pipeline["audio_job_id"]))

    removed_job_ids.update(
        job_id for job_id, job in jobs.items() if job.get("session_id") == session_id
    )
    for job_id in removed_job_ids:
        jobs.pop(job_id, None)

    shutil.rmtree(session_dir)

    return {
        "session_id": session_id,
        "deleted": True,
        "removed_jobs": len(removed_job_ids),
    }


@app.get("/api/jobs")
async def get_all_job_statuses():
    return jobs


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/pipelines")
async def get_all_pipeline_statuses():
    return pipeline_jobs


@app.get("/api/pipelines/{pipeline_job_id}")
async def get_pipeline_status(pipeline_job_id: str):
    pipeline = pipeline_jobs.get(pipeline_job_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline job not found")
    return pipeline


@app.get("/api/sessions/{session_id}/status")
async def get_session_pipeline_status(session_id: str):
    pipeline = pipeline_jobs.get(session_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Session pipeline not found")
    return _pipeline_snapshot_payload(pipeline)


def _read_csv_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_float_or_keep(value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return value


def _normalize_commentary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    numeric_fields = {"start_time", "end_time", "timestamp"}
    for row in rows:
        parsed = dict(row)
        for field in numeric_fields:
            if field in parsed:
                parsed[field] = _parse_float_or_keep(parsed.get(field))
        normalized.append(parsed)
    return normalized


def _resolve_session_state(session_id: str) -> dict[str, Any]:
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists() or not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")

    commentary_path = session_dir / "commentary.csv"
    audio_path = session_dir / "commentary.wav"
    telemetry_path = session_dir / "telemetry.csv"

    if audio_path.exists():
        status = JobStatus.DONE.value
    elif commentary_path.exists():
        status = JobStatus.GENERATING_AUDIO_COMMENTARY.value
    elif (session_dir / "events.csv").exists():
        status = JobStatus.GENERATING_TEXT_COMMENTARY.value
    else:
        status = JobStatus.QUEUED.value

    pipeline = pipeline_jobs.get(session_id)
    if pipeline:
        status = pipeline["status"]

    return {
        "session_id": session_id,
        "status": status,
        "session_dir": str(session_dir),
        "commentary_path": commentary_path,
        "audio_path": audio_path,
        "telemetry_path": telemetry_path,
    }


@app.get("/api/sessions/{session_id}/result")
async def get_session_result(
    session_id: str,
    include_telemetry_csv: bool = False,
    include_events_csv: bool = False,
):
    state = _resolve_session_state(session_id)
    status = state["status"]

    commentary_path: Path = state["commentary_path"]
    audio_path: Path = state["audio_path"]
    telemetry_path: Path = state["telemetry_path"]
    events_path = Path(state["session_dir"]) / "events.csv"

    if (
        status != JobStatus.DONE.value
        or not commentary_path.exists()
        or not audio_path.exists()
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Session generation is not complete yet. " f"Current status: {status}"
            ),
        )

    commentary_rows = _normalize_commentary_rows(_read_csv_rows(commentary_path))
    audio_bytes = audio_path.read_bytes()
    payload = {
        "session_id": session_id,
        "status": status,
        "commentary": {
            "path": str(commentary_path),
            "rows": commentary_rows,
            "count": len(commentary_rows),
        },
        "audio": {
            "path": str(audio_path),
            "mime_type": "audio/wav",
            "bytes": len(audio_bytes),
            "base64": base64.b64encode(audio_bytes).decode("ascii"),
        },
    }

    if include_telemetry_csv:
        if not telemetry_path.exists():
            payload["telemetry"] = None
        else:
            telemetry_rows = _read_csv_rows(telemetry_path)
            payload["telemetry"] = {
                "path": str(telemetry_path),
                "rows": telemetry_rows,
                "count": len(telemetry_rows),
            }

    if include_events_csv:
        if not events_path.exists():
            payload["events"] = None
        else:
            event_rows = _read_csv_rows(events_path)
            payload["events"] = {
                "path": str(events_path),
                "rows": event_rows,
                "count": len(event_rows),
            }

    return payload


@app.post("/api/sessions/{id}/generate")
async def call_llm(req: LLMReq):
    out = llm(
        req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return {"text": out["choices"][0]["text"]}
