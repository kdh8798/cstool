import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.run_pipeline import run_pipeline


# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent

# UI 브랜치의 웹 파일 위치
UI_DIR = BASE_DIR / "webapp"

# 업로드 오디오 임시 저장 위치
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# FastAPI 앱 생성
app = FastAPI(
    title="CSTOOL Whisper LoRA API",
    description="Audio upload → Whisper LoRA ASR → Feedback",
    version="1.0.0",
)


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# UI 정적 파일 서빙
if UI_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=UI_DIR),
        name="static",
    )


@app.get("/")
async def serve_index():
    """
    웹 앱 메인 화면 제공
    """
    index_path = UI_DIR / "index.html"

    if not index_path.exists():
        return {
            "error": "UI file not found",
            "expected_path": str(index_path),
        }

    return FileResponse(index_path)


@app.get("/index.html")
async def serve_index_html():
    return FileResponse(UI_DIR / "index.html")


@app.get("/chat.html")
async def serve_chat():
    return FileResponse(UI_DIR / "chat.html")


@app.get("/history.html")
async def serve_history():
    return FileResponse(UI_DIR / "history.html")


@app.get("/settings.html")
async def serve_settings():
    return FileResponse(UI_DIR / "settings.html")


@app.get("/wordbook.html")
async def serve_wordbook():
    return FileResponse(UI_DIR / "wordbook.html")


@app.get("/health")
async def health_check():
    """
    서버 상태 확인용
    """
    return {
        "status": "ok",
        "ui_dir": str(UI_DIR),
        "ui_exists": UI_DIR.exists(),
        "upload_dir": str(UPLOAD_DIR),
    }


# 음성 인식 API
@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    오디오 파일 업로드 후 Whisper + LoRA 추론 및 피드백 반환
    """
    safe_filename = file.filename or "uploaded_audio.webm"
    save_path = UPLOAD_DIR / safe_filename

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = run_pipeline(
        audio_path=save_path,
        language="auto",
    )

    return {
        "transcription": result.get("transcription", ""),
        "raw_transcription": result.get("raw_transcription", ""),
        "asr_candidates": result.get("asr_candidates", {}),
        "rule_feedback": result.get("rule_feedback", ""),
        "feedback": result.get("feedback", ""),
        "audio_path": str(save_path),
        "lora_path": result.get("lora_path", ""),
        "analysis": result.get("analysis", {}),
        "scored_candidates": result.get("scored_candidates", []),
    }
