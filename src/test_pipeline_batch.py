import json
from pathlib import Path
from datetime import datetime

from run_pipeline import run_pipeline


BASE_DIR = Path(__file__).resolve().parent.parent

SAMPLES_DIR = BASE_DIR / "data" / "samples"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "pipeline_batch_outputs.json"

AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".flac"]


def guess_language(audio_path: Path) -> str:
    """
    파일명 기준으로 language 옵션을 자동 추정한다.
    - ko 포함: 한국어
    - ru 포함: 러시아어
    - cs 또는 codeswitch 포함: 자동 감지
    """
    name = audio_path.stem.lower()

    if "codeswitch" in name or "cs" in name:
        return "auto"

    if "ko" in name or "kor" in name or "korean" in name:
        return "ko"

    if "ru" in name or "rus" in name or "russian" in name:
        return "ru"

    return "auto"


def collect_audio_files():
    if not SAMPLES_DIR.exists():
        raise FileNotFoundError(f"샘플 폴더를 찾지 못했습니다: {SAMPLES_DIR}")

    audio_files = []

    for path in SAMPLES_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(path)

    audio_files.sort()

    return audio_files


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = collect_audio_files()

    if not audio_files:
        print(f"[WARN] 오디오 파일이 없습니다: {SAMPLES_DIR}")
        return

    print("========== Batch Pipeline Test ==========")
    print(f"Samples dir: {SAMPLES_DIR}")
    print(f"Found audio files: {len(audio_files)}")
    print("=========================================\n")

    batch_results = []

    for idx, audio_path in enumerate(audio_files, start=1):
        language = guess_language(audio_path)

        print(f"\n[{idx}/{len(audio_files)}] Running pipeline")
        print(f"Audio: {audio_path.name}")
        print(f"Language: {language}")

        try:
            result = run_pipeline(
                audio_path=audio_path,
                language=language
            )

            batch_results.append({
                "index": idx,
                "status": "success",
                "audio_path": str(audio_path),
                "audio_name": audio_path.name,
                "language": language,
                "transcription": result.get("transcription", ""),
                "feedback": result.get("feedback", ""),
                "lora_path": result.get("lora_path", ""),
            })

        except Exception as e:
            print(f"[ERROR] {audio_path.name} | {type(e).__name__}: {e}")

            batch_results.append({
                "index": idx,
                "status": "error",
                "audio_path": str(audio_path),
                "audio_name": audio_path.name,
                "language": language,
                "error_type": type(e).__name__,
                "error_message": str(e),
            })

    final_output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "samples_dir": str(SAMPLES_DIR),
        "num_files": len(audio_files),
        "num_success": sum(1 for item in batch_results if item["status"] == "success"),
        "num_error": sum(1 for item in batch_results if item["status"] == "error"),
        "results": batch_results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print("\n========== Batch Test Complete ==========")
    print(f"Total: {final_output['num_files']}")
    print(f"Success: {final_output['num_success']}")
    print(f"Error: {final_output['num_error']}")
    print(f"Saved to: {OUTPUT_JSON}")
    print("=========================================")


if __name__ == "__main__":
    main()
