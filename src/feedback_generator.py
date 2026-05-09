import json
import re
from pathlib import Path


CYRILLIC_PATTERN = re.compile(r"[\u0400-\u04FF]")
KEYWORD_FILE = Path(__file__).with_name("math_keywords.json")

DEFAULT_KEYWORD_EXPLANATIONS = {
    "дробь": "러시아어로 '분수'",
    "числитель": "러시아어로 '분자'",
    "знаменатель": "러시아어로 '분모'",
    "분수": "전체를 여러 부분으로 나눈 값 중 일부",
    "분자": "분수에서 위에 있는 수",
    "분모": "분수에서 아래에 있는 수",
}

DEFAULT_FEEDBACK = "학습 피드백: 코드스위칭이나 주요 학습 키워드가 감지되지 않았습니다."


def load_keyword_explanations(keyword_file: Path = KEYWORD_FILE) -> dict[str, str]:
    """수학 키워드 설명 파일을 읽는다. 파일이 없으면 기본 키워드를 사용한다."""
    if not keyword_file.exists():
        return DEFAULT_KEYWORD_EXPLANATIONS

    with keyword_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"키워드 파일은 JSON 객체 형식이어야 합니다: {keyword_file}")

    return {str(keyword): str(explanation) for keyword, explanation in data.items()}


def generate_feedback(text: str) -> str:
    """Whisper STT 텍스트에서 간단한 학습 피드백을 생성한다."""
    feedback = []
    keyword_explanations = load_keyword_explanations()

    if CYRILLIC_PATTERN.search(text):
        feedback.append("코드스위칭 감지: 러시아어 키릴 문자가 포함되어 있습니다.")

    for keyword, explanation in keyword_explanations.items():
        if keyword in text:
            feedback.append(f"{keyword}: {explanation}")

    if not feedback:
        return DEFAULT_FEEDBACK

    return "\n".join(feedback)


# CLI 테스트용
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python feedback_generator.py \"피드백을 생성할 텍스트\"")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    print(generate_feedback(input_text))
