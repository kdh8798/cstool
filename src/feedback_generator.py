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

DEFAULT_FEEDBACK = (
    "학습 피드백: 등록된 러시아어 학습 키워드가 뚜렷하게 감지되지 않았습니다. "
    "인식된 문장을 확인하고, 필요한 단어나 표현을 다시 말해 보세요."
)

RUSSIAN_TERM_EXPLANATIONS = {
    "дробь": "러시아어로 '분수'",
    "знаменатель": "러시아어로 '분모'",
    "числитель": "러시아어로 '분자'",
    "плюс": "러시아어로 '더하기'",
    "минус": "러시아어로 '빼기'",
    "равно": "러시아어로 '같다'",
    "число": "러시아어로 '숫자'",
    "ноль": "러시아어로 '0 또는 영'",
    "один": "러시아어로 '1 또는 일'",
    "два": "러시아어로 '2 또는 이'",
    "три": "러시아어로 '3 또는 삼'",
    "пять": "러시아어로 '5 또는 오'",
    "десять": "러시아어로 '10 또는 십'",
    "квадрат": "러시아어로 '정사각형 또는 네모'",
    "круг": "러시아어로 '원 또는 동그라미'",
    "треугольник": "러시아어로 '삼각형 또는 세모'",
    "линия": "러시아어로 '선'",
    "угол": "러시아어로 '각'",
    "сантиметр": "러시아어로 '센티미터'",
    "метр": "러시아어로 '미터'",
    "ответ": "러시아어로 '답'",
    "задача": "러시아어로 '문제'",
    "пример": "러시아어로 '예제'",
    "сумма": "러시아어로 '합 또는 더한 값'",
    "разность": "러시아어로 '차 또는 뺀 값'",
    "остаток": "러시아어로 '나머지'",
    "цифра": "러시아어로 '숫자 또는 한 자리 숫자'",
    "чётное число": "러시아어로 '짝수'",
    "нечётное число": "러시아어로 '홀수'",
    "больше": "러시아어로 '더 크다'",
    "меньше": "러시아어로 '더 작다'",
    "умножить": "러시아어로 '곱하기'",
    "разделить": "러시아어로 '나누기'",
    "я": "러시아어로 '나는'",
    "ты": "러시아어로 '너는'",
    "он": "러시아어로 '그는'",
    "она": "러시아어로 '그녀는'",
    "это": "러시아어로 '이것'",
    "что": "러시아어로 '무엇'",
    "как": "러시아어로 '어떻게'",
    "где": "러시아어로 '어디'",
    "почему": "러시아어로 '왜'",
    "сколько": "러시아어로 '얼마나 또는 몇 개'",
    "хочу": "러시아어로 '원하다'",
    "знаю": "러시아어로 '알다'",
    "не знаю": "러시아어로 '모르다'",
    "правильно": "러시아어로 '맞다 또는 올바르다'",
    "ошибка": "러시아어로 '오류 또는 실수'",
}

RUSSIAN_TERM_ALIASES = {
    "дробь": ["дробь", "дроби", "드롭", "드로비", "드로브"],
    "знаменатель": ["знаменатель", "즈나몌나텔", "즈나메나텔", "즈나미나텔"],
    "числитель": ["числитель", "치슬리텔", "치슬리쩰"],
    "плюс": ["плюс", "플류스", "플러스"],
    "минус": ["минус", "미누스", "마이너스"],
    "равно": ["равно", "라브나", "라브노"],
    "число": ["число", "치슬로"],
    "ноль": ["ноль", "놀", "노리"],
    "один": ["один", "아진", "오딘"],
    "два": ["два", "드바"],
    "три": ["три", "트리"],
    "пять": ["пять", "퍄티", "퍄트"],
    "десять": ["десять", "제샤티", "데샤티"],
    "квадрат": ["квадрат", "크바드라트"],
    "круг": ["круг", "크룩", "크루그"],
    "треугольник": ["треугольник", "트리우골닉", "트레우골닉"],
    "линия": ["линия", "리니야"],
    "угол": ["угол", "우골"],
    "сантиметр": ["сантиметр", "산치메트르", "센치메트르"],
    "метр": ["метр", "메트르"],
    "ответ": ["ответ", "아트볫", "아트벳"],
    "задача": ["задача", "자다차"],
    "пример": ["пример", "프리메르"],
    "сумма": ["сумма", "숨마"],
    "разность": ["разность", "라즈노스티"],
    "остаток": ["остаток", "아스타탁", "오스타탁"],
    "цифра": ["цифра", "치프라"],
    "чётное число": ["чётное число", "четное число", "촛나예 치슬로"],
    "нечётное число": ["нечётное число", "нечетное число", "니촛나예 치슬로"],
    "больше": ["больше", "볼셰"],
    "меньше": ["меньше", "몐셰", "멘셰"],
    "умножить": ["умножить", "움노지트"],
    "разделить": ["разделить", "라즈젤리트", "라즈델리트"],
    "я": ["я", "야"],
    "ты": ["ты", "띄", "티"],
    "это": ["это", "에따", "에타"],
    "что": ["что", "쉬또", "슈토"],
    "как": ["как", "까끄", "카크"],
    "где": ["где", "그제", "그데"],
    "почему": ["почему", "빠치무", "파치무"],
    "сколько": ["сколько", "스콜카"],
    "хочу": ["хочу", "하추", "하츄", "호추"],
    "знаю": ["знаю", "즈나유"],
    "не знаю": ["не знаю", "네즈나유"],
    "правильно": ["правильно", "프라빌나"],
    "ошибка": ["ошибка", "아쉽카", "오쉽카"],
}

INTENT_RULES = [
    {
        "term": "дробь",
        "markers": ["밑", "아래", "하단"],
        "feedback": [
            "질문 의도: 분수에서 아래에 있는 숫자의 한국어 명칭을 묻고 있음",
            "정답: 분모",
        ],
    },
    {
        "term": "дробь",
        "markers": ["위", "위쪽", "상단"],
        "feedback": [
            "질문 의도: 분수에서 위에 있는 숫자의 한국어 명칭을 묻고 있음",
            "정답: 분자",
        ],
    },
]


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
    normalized_text = text.casefold()
    matched_terms = set()

    has_alias = any(
        alias.casefold() in normalized_text
        for aliases in RUSSIAN_TERM_ALIASES.values()
        for alias in aliases
    )

    if CYRILLIC_PATTERN.search(text) or has_alias:
        feedback.append("코드스위칭 감지: 러시아어 용어 또는 키릴 문자가 포함되어 있습니다.")

    for canonical, aliases in RUSSIAN_TERM_ALIASES.items():
        if any(alias.casefold() in normalized_text for alias in aliases):
            explanation = keyword_explanations.get(
                canonical,
                RUSSIAN_TERM_EXPLANATIONS.get(canonical, "러시아어 수학 용어입니다."),
            )
            feedback.append(f"{canonical}: {explanation}")
            matched_terms.add(canonical)

    for keyword, explanation in keyword_explanations.items():
        normalized_keyword = keyword.casefold()
        if keyword in matched_terms:
            continue
        if normalized_keyword in normalized_text:
            feedback.append(f"{keyword}: {explanation}")
            matched_terms.add(keyword)

    for rule in INTENT_RULES:
        if rule["term"] in matched_terms and any(marker in text for marker in rule["markers"]):
            feedback.extend(rule["feedback"])

    if not feedback:
        return DEFAULT_FEEDBACK

    return "\n".join(feedback)


# analysis 기반 피드백 생성 구조
def generate_feedback_from_analysis(analysis: dict) -> str:
    tokens = analysis.get("tokens", [])
    corrected_text = analysis.get("corrected_text", "")
    summary = analysis.get("summary", "")

    ru_tokens = [
        token for token in tokens
        if token.get("language") == "ru"
    ]

    uncertain_tokens = [
        token for token in tokens
        if token.get("language") == "uncertain"
    ]

    feedback = []

    if ru_tokens:
        feedback.append("코드스위칭 감지: 러시아어 표현이 포함되어 있습니다.")

        for token in ru_tokens:
            text = token.get("text", "")
            meaning = token.get("meaning", "")
            confidence = token.get("confidence", "")

            if meaning:
                feedback.append(f"{text}: {meaning}")
            else:
                feedback.append(f"{text}: 러시아어 표현으로 감지되었습니다.")

    elif uncertain_tokens:
        feedback.append("코드스위칭 후보 감지: 러시아어처럼 들리는 표현이 있습니다.")

        for token in uncertain_tokens:
            text = token.get("text", "")
            feedback.append(f"{text}: 러시아어 표현일 가능성이 있어 문맥 확인이 필요합니다.")

    else:
        feedback.append(
            "등록된 러시아어 표현이 뚜렷하게 감지되지 않았습니다. "
            "인식된 문장을 확인하고 필요한 표현을 다시 말해 보세요."
        )

    if summary:
        feedback.append(f"분석 요약: {summary}")

    if corrected_text:
        feedback.append(f"복원 문장: {corrected_text}")

    return "\n".join(feedback)


# CLI 테스트용
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python feedback_generator.py \"피드백을 생성할 텍스트\"")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    print(generate_feedback(input_text))
