import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

RESULT_JSON = BASE_DIR / "results" / "pipeline_batch_outputs.json"
SUMMARY_TXT = BASE_DIR / "results" / "pipeline_batch_summary.txt"


def shorten(text: str, max_len: int = 60) -> str:
    if text is None:
        return ""

    text = str(text).replace("\n", " ").strip()

    if len(text) <= max_len:
        return text

    return text[:max_len] + "..."


def load_results():
    if not RESULT_JSON.exists():
        raise FileNotFoundError(
            f"결과 파일을 찾지 못했습니다: {RESULT_JSON}\n"
            "먼저 python src/test_pipeline_batch.py 를 실행하세요."
        )

    with open(RESULT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def build_summary(data):
    results = data.get("results", [])

    total = data.get("num_files", len(results))
    success = data.get("num_success", 0)
    error = data.get("num_error", 0)

    lines = []

    lines.append("========== Pipeline Batch Summary ==========")
    lines.append(f"Total files : {total}")
    lines.append(f"Success     : {success}")
    lines.append(f"Error       : {error}")
    lines.append("============================================")
    lines.append("")

    lines.append("파일별 결과")
    lines.append("-" * 100)
    lines.append(f"{'File':25} | {'Lang':8} | {'Status':8} | {'Transcription':30} | {'Feedback'}")
    lines.append("-" * 100)

    for item in results:
        audio_name = item.get("audio_name", "")
        language = item.get("language", "")
        status = item.get("status", "")

        if status == "success":
            transcription = shorten(item.get("transcription", ""), 30)
            feedback = shorten(item.get("feedback", ""), 30)
        else:
            transcription = "-"
            feedback = shorten(
                f"{item.get('error_type', '')}: {item.get('error_message', '')}",
                30
            )

        lines.append(
            f"{audio_name:25} | {language:8} | {status:8} | "
            f"{transcription:30} | {feedback}"
        )

    lines.append("-" * 100)
    lines.append("")

    if error > 0:
        lines.append("오류 목록")
        lines.append("-" * 100)

        for item in results:
            if item.get("status") == "error":
                lines.append(
                    f"- {item.get('audio_name')} | "
                    f"{item.get('error_type')} | "
                    f"{item.get('error_message')}"
                )

        lines.append("")

    lines.append("피드백 자료용 요약 문장")
    lines.append("-" * 100)

    if error == 0:
        lines.append(
            f"총 {total}개의 테스트 오디오에 대해 파이프라인을 실행한 결과, "
            f"모든 샘플에서 음성 인식 및 피드백 생성이 정상적으로 수행되었다."
        )
    else:
        lines.append(
            f"총 {total}개의 테스트 오디오 중 {success}개는 정상 처리되었고, "
            f"{error}개는 오류가 발생하였다. 오류 샘플에 대해서는 파일 경로, "
            f"오디오 품질, 모델 로딩 여부를 추가 확인할 예정이다."
        )

    return "\n".join(lines)


def main():
    data = load_results()
    summary = build_summary(data)

    print(summary)

    SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\nSaved summary to: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
