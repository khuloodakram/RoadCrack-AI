"""Bounded upload API. Deploy behind authentication before public use."""
import logging
import os
import shutil
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from roadcrack.exports import image_exports, video_exports
from roadcrack.pipeline import Pipeline, WeightsUnavailable, registry
from roadcrack.video import Cancelled, process_video

app = FastAPI(title="RoadCrack AI", version="0.1.0")
MAX_BODY = 100 * 1024**2
VIDEO_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="roadcrack")
VIDEO_SLOTS = threading.BoundedSemaphore(2)
IMAGE_SLOTS = threading.BoundedSemaphore(2)
JOBS, JOB_LOCK = {}, threading.Lock()
FILES = {"original.png", "overlay.png", "mask.png", "analysis.json", "detections.csv", "report.pdf",
         "annotated.mp4", "frames.jsonl"}
LOG = logging.getLogger(__name__)


class UploadLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        from starlette.responses import JSONResponse
        async def reject():
            await JSONResponse({"detail": "الملف أكبر من الحد المسموح"}, 413)(scope, receive, send)

        length = dict(scope["headers"]).get(b"content-length")
        if length and length.isdigit() and int(length) > MAX_BODY:
            await reject()
            return
        received = 0
        exceeded = False

        async def limited_receive():
            nonlocal received, exceeded
            message = await receive()
            received += len(message.get("body", b""))
            if received > MAX_BODY:
                exceeded = True
                # Starlette closes spooled parts when the multipart parser raises.
                from starlette.formparsers import MultiPartException
                raise MultiPartException("Upload exceeds the request body limit")
            return message

        async def limited_send(message):
            if not exceeded:
                await send(message)

        await self.app(scope, limited_receive, limited_send)
        if exceeded:
            await reject()



app.add_middleware(UploadLimit)


def root():
    return Path(os.getenv("ROADCRACK_ARTIFACTS", "artifacts")).resolve()


def save_upload(upload, target, limit):
    count = 0
    with target.open("wb") as stream:
        while chunk := upload.file.read(1024**2):
            count += len(chunk)
            if count > limit:
                raise HTTPException(413, "الملف أكبر من الحد المسموح")
            stream.write(chunk)
    if not count:
        raise HTTPException(422, "الملف فارغ")


def create_pipeline(model, mode, confidence):
    try:
        return Pipeline(model, mode, confidence)
    except WeightsUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        LOG.exception("Model load failed")
        raise HTTPException(503, "تعذر تحميل النموذج. تحقق من الأوزان وإعدادات الجهاز") from exc


@app.get("/health")
def health():
    return {"status": "ok", "models": list(registry())}


@app.post("/images")
def images(file: UploadFile = File(), model: str = Form("yolo26s"),
           mode: str = Form("fast"), confidence: float = Form(.25)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "نوع الصورة غير مدعوم")
    if not IMAGE_SLOTS.acquire(blocking=False):
        raise HTTPException(429, "المعالجة مشغولة الآن")
    job = uuid4()
    folder = root() / str(job)
    try:
        folder.mkdir(parents=True)
        uploaded = folder / "upload"
        save_upload(file, uploaded, 15 * 1024**2)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(uploaded) as opened:
                    if opened.format not in {"JPEG", "PNG", "WEBP"} or opened.width * opened.height > 24_000_000:
                        raise ValueError("Unsupported dimensions or format")
                    image = ImageOps.exif_transpose(opened).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
                Image.DecompressionBombWarning) as exc:
            raise HTTPException(422, "الصورة تالفة أو تتجاوز حدود الأبعاد") from exc
        uploaded.unlink()
        analysis = create_pipeline(model, mode, confidence).analyze(image)
        image_exports(folder, image, analysis)
        return {"id": job, **analysis.result}
    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        LOG.exception("Image processing failed")
        raise HTTPException(500, "تعذر إكمال تحليل الصورة") from exc
    finally:
        IMAGE_SLOTS.release()


def video_worker(job, folder, path, options, sample, event):
    try:
        if event.is_set():
            raise Cancelled()
        with JOB_LOCK:
            JOBS[job]["status"] = "running"
        pipeline = create_pipeline(*options)
        summary = process_video(path, folder, pipeline, sample, event.is_set)
        if event.is_set():
            raise Cancelled()
        video_exports(folder, summary)
        result = {"status": "done", **summary}
    except Cancelled:
        result = {"status": "cancelled"}
    except (HTTPException, ValueError) as exc:
        result = {"status": "failed", "detail": exc.detail if isinstance(exc, HTTPException) else str(exc)}
    except Exception:
        LOG.exception("Video processing failed")
        result = {"status": "failed", "detail": "تعذر إكمال معالجة الفيديو"}
    path.unlink(missing_ok=True)
    if result["status"] != "done":
        shutil.rmtree(folder, ignore_errors=True)
    with JOB_LOCK:
        JOBS[job].update(result)
    VIDEO_SLOTS.release()


@app.post("/videos", status_code=202)
def videos(file: UploadFile = File(), model: str = Form("yolo26s"), mode: str = Form("fast"),
           confidence: float = Form(.25), sample: int = Form(5, ge=1, le=120)):
    if file.content_type not in {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm",
                                 "video/x-matroska", "video/avi"}:
        raise HTTPException(415, "نوع الفيديو غير مدعوم")
    if not VIDEO_SLOTS.acquire(False):
        raise HTTPException(429, "قائمة معالجة الفيديو ممتلئة")
    job = uuid4()
    folder = root() / str(job)
    try:
        folder.mkdir(parents=True)
        path = folder / "upload"
        save_upload(file, path, 95 * 1024**2)
        event = threading.Event()
        with JOB_LOCK:
            JOBS[job] = {"status": "queued", "event": event}
        VIDEO_WORKER.submit(video_worker, job, folder, path, (model, mode, confidence), sample, event)
    except Exception:
        VIDEO_SLOTS.release()
        shutil.rmtree(folder, ignore_errors=True)
        with JOB_LOCK:
            JOBS.pop(job, None)
        raise
    return {"id": job, "status": "queued"}


@app.get("/videos/{job}")
def video_status(job: UUID):
    with JOB_LOCK:
        if job not in JOBS:
            raise HTTPException(404, "لم يتم العثور على التحليل")
        return {key: value for key, value in JOBS[job].items() if key != "event"}


@app.delete("/videos/{job}")
def stop_video(job: UUID):
    with JOB_LOCK:
        entry = JOBS.get(job)
        if not entry:
            raise HTTPException(404, "لم يتم العثور على التحليل")
        if entry["status"] in {"queued", "running"}:
            entry["event"].set()
            return {"status": "cancelling"}
        return {"status": entry["status"]}


@app.get("/artifacts/{job}/{filename}")
def artifact(job: UUID, filename: str):
    with JOB_LOCK:
        pending = job in JOBS and JOBS[job]["status"] != "done"
    if filename not in FILES or pending:
        raise HTTPException(404, "ملف النتيجة غير متاح")
    path = root() / str(job) / filename
    if not path.is_file():
        raise HTTPException(404, "ملف النتيجة غير متاح")
    return FileResponse(path, filename=filename)
