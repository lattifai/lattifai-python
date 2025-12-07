import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
from dotenv import find_dotenv, load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Try to find and load .env file from current directory or parent directories
load_dotenv(find_dotenv(usecwd=True))


app = FastAPI(title="LattifAI Web Interface")

print(f"LOADING APP FROM: {__file__}")

# Lazy-initialized client - will be created on first use
_client = None


def get_client():
    """Get or create the LattifAI client (lazy initialization)."""
    global _client
    if _client is None:
        from lattifai.client import LattifAI

        _client = LattifAI()
    return _client


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


def mask_api_key(key: str) -> str:
    """Mask API key for display, showing only first 6 and last 4 characters."""
    if len(key) <= 10:
        return "*" * len(key)
    return key[:6] + "*" * (len(key) - 10) + key[-4:]


@app.get("/api/keys")
async def get_api_keys():
    """Get status of API keys from environment variables."""
    lattifai_key = os.environ.get("LATTIFAI_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    return {
        "lattifai": {
            "exists": bool(lattifai_key),
            "masked_value": mask_api_key(lattifai_key) if lattifai_key else None,
            "create_url": "https://lattifai.com/dashboard/api-keys",
        },
        "gemini": {
            "exists": bool(gemini_key),
            "masked_value": mask_api_key(gemini_key) if gemini_key else None,
            "create_url": "https://aistudio.google.com/apikey",
        },
    }


@app.post("/api/keys")
async def save_api_keys(request: Request):
    """Save API keys to environment variables and optionally to .env file."""
    try:
        data = await request.json()
        lattifai_key = data.get("lattifai_key", "").strip()
        gemini_key = data.get("gemini_key", "").strip()
        save_to_file = data.get("save_to_file", False)  # Optional: save to .env file

        # Always update environment variables in current process
        if lattifai_key:
            os.environ["LATTIFAI_API_KEY"] = lattifai_key
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key

        # Reset client to force re-initialization with new keys
        global _client
        _client = None

        result = {
            "status": "success",
            "message": "API keys updated in environment variables",
        }

        # Optionally save to .env file for persistence
        if save_to_file:
            # Find the .env file path
            env_path = find_dotenv(usecwd=True)
            if not env_path:
                # Create .env in current working directory
                env_path = Path.cwd() / ".env"

            # Read existing .env content
            env_lines = []
            if Path(env_path).exists():
                with open(env_path, "r") as f:
                    env_lines = f.readlines()

            # Update or add API keys
            updated_lines = []
            lattifai_updated = False
            gemini_updated = False

            for line in env_lines:
                if line.strip().startswith("LATTIFAI_API_KEY=") or line.strip().startswith("#LATTIFAI_API_KEY="):
                    if lattifai_key:
                        updated_lines.append(f"LATTIFAI_API_KEY={lattifai_key}\n")
                        lattifai_updated = True
                    else:
                        updated_lines.append(line)  # Keep existing or commented out
                elif line.strip().startswith("GEMINI_API_KEY=") or line.strip().startswith("#GEMINI_API_KEY="):
                    if gemini_key:
                        updated_lines.append(f"GEMINI_API_KEY={gemini_key}\n")
                        gemini_updated = True
                    else:
                        updated_lines.append(line)  # Keep existing or commented out
                else:
                    updated_lines.append(line)

            # Add new keys if they weren't in the file
            if lattifai_key and not lattifai_updated:
                updated_lines.append(f"LATTIFAI_API_KEY={lattifai_key}\n")
            if gemini_key and not gemini_updated:
                updated_lines.append(f"GEMINI_API_KEY={gemini_key}\n")

            # Write back to .env file
            with open(env_path, "w") as f:
                f.writelines(updated_lines)

            result["message"] = "API keys saved to environment variables and .env file"
            result["env_path"] = str(env_path)

        return result

    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})


@app.post("/api/utils/select-directory")
async def select_directory():
    """
    Open a native directory selection dialog on the server (local machine).
    Returns the selected path.
    """
    try:
        path = ""
        if sys.platform == "darwin":
            # Use AppleScript for macOS - it's cleaner than Tkinter on Mac
            script = """
            try
                set theFolder to choose folder with prompt "Select Output Directory"
                POSIX path of theFolder
            on error
                return ""
            end try
            """
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            if result.returncode == 0:
                path = result.stdout.strip()

        # Fallback to Tkinter if path is still empty (e.g. not mac or mac script failed)
        # Note: Tkinter might not be installed or might fail in some environments
        if not path and sys.platform != "darwin":
            try:
                import tkinter
                from tkinter import filedialog

                root = tkinter.Tk()
                root.withdraw()  # Hide main window
                root.wm_attributes("-topmost", 1)  # Bring to front
                path = filedialog.askdirectory(title="Select Output Directory")
                root.destroy()
            except ImportError:
                pass
            except Exception as e:
                print(f"Tkinter dialog failed: {e}")

        return {"path": path}
    except Exception as e:
        # Don't fail the request, just return empty path or error logged
        print(f"Directory selection failed: {e}")
        return {"path": "", "error": str(e)}


@app.post("/align")
async def align_files(
    background_tasks: BackgroundTasks,
    media_file: Optional[UploadFile] = File(None),
    caption_file: Optional[UploadFile] = File(None),
    local_media_path: Optional[str] = Form(None),
    local_caption_path: Optional[str] = Form(None),
    local_output_dir: Optional[str] = Form(None),
    youtube_url: Optional[str] = Form(None),
    youtube_output_dir: Optional[str] = Form(None),
    split_sentence: bool = Form(True),
    normalize_text: bool = Form(False),
    output_format: str = Form("srt"),
    transcription_model: str = Form("nvidia/parakeet-tdt-0.6b-v3"),
    alignment_model: str = Form("Lattifai/Lattice-1"),
):
    # Check if LATTIFAI_API_KEY is set
    if not os.environ.get("LATTIFAI_API_KEY"):
        return JSONResponse(
            status_code=400,
            content={
                "error": "LATTIFAI_API_KEY is not set. Please set the environment variable or add it to your .env file.",
                "help_url": "https://lattifai.com/dashboard/api-keys",
            },
        )

    if not media_file and not youtube_url and not local_media_path:
        return JSONResponse(
            status_code=400, content={"error": "Either media file, local media path, or YouTube URL must be provided."}
        )

    # Get lazily initialized client
    client = get_client()
    if not client:
        # This should rarely happen due to lazy init, but just in case
        return JSONResponse(
            status_code=500,
            content={
                "error": "LattifAI client not initialized. Please check API key configuration.",
            },
        )

    media_path = None
    caption_path = None
    temp_files_to_delete = []

    try:
        if media_file:
            # Save uploaded media file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(media_file.filename).suffix) as tmp_media:
                content = await media_file.read()
                tmp_media.write(content)
                media_path = tmp_media.name
                temp_files_to_delete.append(media_path)

            if caption_file:
                # Save uploaded caption file to a temporary location
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(caption_file.filename).suffix
                ) as tmp_caption:
                    content = await caption_file.read()
                    tmp_caption.write(content)
                    caption_path = tmp_caption.name
                    temp_files_to_delete.append(caption_path)

        elif local_media_path:
            media_path = local_media_path
            if not Path(media_path).exists():
                return JSONResponse(status_code=400, content={"error": f"Local media file not found: {media_path}"})

            if local_caption_path:
                caption_path = local_caption_path
                if not Path(caption_path).exists():
                    return JSONResponse(
                        status_code=400, content={"error": f"Local caption file not found: {caption_path}"}
                    )

        # Process in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        result_caption = await loop.run_in_executor(
            None,
            process_alignment,
            media_path,
            youtube_url,
            youtube_output_dir,
            caption_path,
            local_output_dir,
            split_sentence,
            normalize_text,
            transcription_model,
            alignment_model,
            output_format,
        )

        # Convert result to dict with specified output format
        caption_content = result_caption.to_string(format=output_format)

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
            "caption_content": caption_content,
            "output_format": output_format,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": traceback.format_exc()})


def process_alignment(
    media_path,
    youtube_url,
    youtube_output_dir,
    caption_path,
    local_output_dir,
    split_sentence,
    normalize_text,
    transcription_model,
    alignment_model,
    output_format,
):
    """
    Wrapper to call LattifAI client.
    Note: Transcription will be automatically triggered when no caption is provided.
    """
    # Get lazily initialized client
    client = get_client()
    if not client:
        raise RuntimeError("LattifAI client not initialized")

    # Update caption config
    client.caption_config.normalize_text = normalize_text

    # Check if alignment model changed - if so, reinitialize aligner
    if client.aligner.config.model_name != alignment_model:
        print(
            f"Alignment model changed from {client.aligner.config.model_name} to {alignment_model}, reinitializing aligner..."
        )  # noqa: E501
        from lattifai.alignment import Lattice1Aligner

        client.aligner.config.model_name = alignment_model
        client.aligner = Lattice1Aligner(config=client.aligner.config)

    # Check if transcription model changed - if so, reinitialize transcriber
    if transcription_model != client.transcription_config.model_name:
        print(
            f"Transcription model changed from {client.transcription_config.model_name} to {transcription_model}, reinitializing transcriber..."
        )  # noqa: E501
        from lattifai.config import TranscriptionConfig

        client.transcription_config = TranscriptionConfig(model_name=transcription_model)
        client._transcriber = None

    if youtube_url:
        # If youtube, we use client.youtube
        # Note: client.youtube handles download + alignment
        # Will try to download YT captions first, if not available, will transcribe

        # Determine output directory
        # Default: ~/Downloads/YYYY-MM-DD
        if not youtube_output_dir or not youtube_output_dir.strip():
            from datetime import datetime

            today = datetime.now().strftime("%Y-%m-%d")
            youtube_output_dir = f"~/Downloads/{today}"

        temp_path = Path(youtube_output_dir).expanduser()
        temp_path.mkdir(parents=True, exist_ok=True)

        result = client.youtube(
            url=youtube_url,
            output_dir=temp_path,
            use_transcription=False,  # Try to download captions first
            force_overwrite=True,  # No user prompt in server mode
            split_sentence=split_sentence,
        )
        return result
    else:
        # Local file alignment
        output_caption_path = None
        if local_output_dir:
            output_dir = Path(local_output_dir).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(media_path).stem
            # Prevent overwriting input if names clash, use _LattifAI suffix
            output_filename = f"{stem}_LattifAI.{output_format}"
            output_caption_path = output_dir / output_filename
            print(f"Saving alignment result to: {output_caption_path}")

        # If no caption_path provided, client.alignment will automatically call _transcribe
        return client.alignment(
            input_media=str(media_path),
            input_caption=str(caption_path) if caption_path else None,
            output_caption_path=str(output_caption_path) if output_caption_path else None,
            split_sentence=split_sentence,
        )
