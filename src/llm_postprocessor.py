import os
from typing import Callable, Optional


SYSTEM_PROMPT = (
    "너는 한국어와 러시아어 코드스위칭 상황을 돕는 수학 학습 피드백 도우미다. "
    "학생이 이해하기 쉬운 한국어로 짧고 자연스럽게 피드백한다. "
    "규칙 기반 피드백에 있는 용어 설명과 코드스위칭 감지 내용은 빠뜨리지 않는다. "
    "코드스위칭을 고치거나 줄이라고 말하지 않고, 학습 단서로만 안내한다. "
    "같은 입력에는 가능한 한 같은 문장 구조와 표현을 사용한다."
)


def build_feedback_prompt(stt_text: str, rule_feedback: str) -> str:
    """LLM에 전달할 후처리 프롬프트를 만든다."""
    return (
        "아래 STT 결과와 규칙 기반 피드백을 바탕으로 학생에게 줄 최종 피드백을 작성해줘.\n\n"
        f"[STT 결과]\n{stt_text}\n\n"
        f"[규칙 기반 피드백]\n{rule_feedback}\n\n"
        "조건:\n"
        "- 한국어로 작성한다.\n"
        "- 전체는 정확히 3문장으로 작성한다.\n"
        "- 첫 문장은 코드스위칭 감지 내용을 평가 없이 설명한다.\n"
        "- 두 번째 문장은 러시아어 용어와 한국어 의미를 설명한다.\n"
        "- 마지막 문장은 수학 개념을 다시 연결해 보라는 학습 안내로 마무리한다.\n"
        "- 같은 입력에는 가능한 한 같은 문장 구조와 표현을 사용한다.\n"
        "- '잘했어요', '좋습니다', '한국어로만', '러시아어 대신', '다음에는'이라는 표현은 쓰지 않는다.\n"
        "- 코드스위칭을 고치거나 줄이라고 말하지 않는다."
    )


def fallback_postprocess_feedback(stt_text: str, rule_feedback: str) -> str:
    """LLM API를 쓰지 못할 때 사용할 로컬 후처리 결과를 만든다."""
    if not stt_text.strip():
        return "STT 결과가 비어 있어 피드백을 만들 수 없습니다. 음성 인식 결과를 먼저 확인해 주세요."

    if "감지되지 않았습니다" in rule_feedback:
        return (
            "이번 STT 결과에서는 코드스위칭이나 주요 수학 용어가 뚜렷하게 보이지 않았습니다. "
            "다음에는 수업의 핵심 개념이 잘 드러나도록 다시 말해 보세요."
        )

    details = " ".join(line.strip() for line in rule_feedback.splitlines() if line.strip())
    return (
        "STT 결과에서 학습에 도움이 되는 단서가 확인되었습니다. "
        f"{details} "
        "이 내용을 바탕으로 감지된 수학 용어의 의미를 다시 정리해 보세요."
    )


def _postprocess_with_openai(stt_text: str, rule_feedback: str, model: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_feedback_prompt(stt_text, rule_feedback)},
            ],
            temperature=0,
        )
    except Exception:
        return None
    return response.choices[0].message.content.strip()


def postprocess_feedback(
    stt_text: str,
    rule_feedback: str,
    *,
    use_llm: bool = True,
    model: str = "gpt-4o-mini",
    llm_client: Optional[Callable[[str, str], str]] = None,
) -> str:
    """STT 결과와 규칙 기반 피드백을 자연스러운 최종 학습 피드백으로 다듬는다."""
    if llm_client is not None:
        return llm_client(stt_text, rule_feedback).strip()

    if use_llm:
        llm_feedback = _postprocess_with_openai(stt_text, rule_feedback, model)
        if llm_feedback:
            return llm_feedback

    return fallback_postprocess_feedback(stt_text, rule_feedback)
