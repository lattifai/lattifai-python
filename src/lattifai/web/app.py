import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
from dotenv import find_dotenv, load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from lattifai.caption import Caption
from lattifai.client import LattifAI
from lattifai.config import AlignmentConfig, CaptionConfig

# Try to find and load .env file from current directory or parent directories
load_dotenv(find_dotenv(usecwd=True))


app = FastAPI(title="LattifAI Web Interface")

print(f"LOADING APP FROM: {__file__}")


@app.on_event("startup")
async def startup_event():
    print("Listing all registered routes:")
    for route in app.routes:
        print(f"Route: {route.path} - {route.name}")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"INCOMING REQUEST: {request.method} {request.url}")
    response = await call_next(request)
    print(f"OUTGOING RESPONSE: {response.status_code}")
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LattifAI client (globally or per request?)
# For now, global client is fine for local tool
client = LattifAI()


@app.post("/align")
async def align_files(
    background_tasks: BackgroundTasks,
    media_file: Optional[UploadFile] = File(None),
    caption_file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    split_sentence: bool = Form(True),
    is_transcription: bool = Form(False),
):
    if not media_file and not youtube_url:
        return JSONResponse(status_code=400, content={"error": "Either media file or YouTube URL must be provided."})

    try:
        # Create a unique task ID (simple timestamp based or uuid)
        # For simplicity in this synchronous-ish wrapper, we'll just process immediately
        # but in a real app better to offload to queue.
        # Since this is a local tool, blocking for a bit is acceptable, but let's try to be async where possible.

        # Use temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            media_path = None
            caption_path = None

            if media_file:
                media_path = temp_path / media_file.filename
                with open(media_path, "wb") as buffer:
                    shutil.copyfileobj(media_file.file, buffer)

            if caption_file:
                caption_path = temp_path / caption_file.filename
                with open(caption_path, "wb") as buffer:
                    shutil.copyfileobj(caption_file.file, buffer)

            # Process in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            result_caption = await loop.run_in_executor(
                None, process_alignment, media_path, youtube_url, caption_path, split_sentence, is_transcription
            )

            # Convert result to dict (SRT format text + list of segments)
            return {
                "status": "success",
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "speaker": seg.speaker if hasattr(seg, "speaker") else None,
                    }
                    for seg in result_caption.alignments
                ],
                "srt_content": result_caption.to_string(format="srt"),
            }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})


def process_alignment(media_path, youtube_url, caption_path, split_sentence, is_transcription):
    """
    Wrapper to call LattifAI client.
    """
    if youtube_url:
        # If youtube, we use client.youtube
        # Note: client.youtube handles download + alignment
        # We need a temporary output dir
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            result = client.youtube(
                url=youtube_url,
                output_dir=temp_path,
                use_transcription=is_transcription,
                split_sentence=split_sentence,
            )
            return result
    else:
        # Local file alignment
        # If no caption path and is_transcription is True, then we use ASR?
        # The current client.alignment implementation calls _transcribe if input_caption is None.

        # If user didn't provide caption_file, we pass None to input_caption
        # client.alignment logic: if not input_caption -> _transcribe

        return client.alignment(
            input_media=str(media_path),
            input_caption=str(caption_path) if caption_path else None,
            split_sentence=split_sentence,
        )
