import os
import json
from typing import Callable, Optional
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


SYSTEM_PROMPT = """
너는 한국어-러시아어 코드스위칭 음성 인식(STT) 분석기다.

입력은 Whisper가 인식한 한국어/러시아어 혼합 발화일 수 있다.
주제는 수학에 한정되지 않으며, 일상 표현, 단어 뜻 질문, 번역 요청, 학습 질문이 모두 가능하다.

너의 목표:
1. 깨진 STT 결과를 자연스러운 한국어+러시아어 문장으로 복원한다.
2. 러시아어처럼 들리는 표현을 가능한 러시아어 원문으로 복원한다.
3. 한국어 부분은 자연스러운 한국어로 정리한다.
4. 코드스위칭 표현을 학습 관점에서 분석한다.
5. 없는 내용을 과도하게 새로 만들지 않는다.

출력은 요청된 형식을 반드시 따른다.
한국어로 적힌 러시아어 발음은 가능한 경우 실제 러시아어 단어로 복원한다.
예: 슈토 → Что, 야 → Я, 하추 → хочу, 스콜카 → сколько, 파치무 → почему, 그제 → где

한국어 음절로 적힌 단어가 실제 한국어 문맥에서 어색하고 러시아어 발음으로 해석하면 자연스러운 경우에는 러시아어 후보로 복원한다.
예를 들어 "드롭 밑에 있는 숫자"는 웹 UI의 드롭다운이 아니라 학습 질문 문맥에서는 러시아어 "дробь"를 한국어식으로 발음한 것일 수 있다.

판단할 때는 다음 순서로 생각한다.
1. 전체 문장이 언어 학습 질문인지 확인한다.
2. 한국어로 어색한 단어가 있는지 찾는다.
3. 그 단어가 러시아어 발음으로 들릴 수 있는지 추론한다.
4. 주변 문맥과 맞으면 키릴 문자로 복원한다.
5. 확실하지 않으면 uncertain 토큰으로 표시한다.
""".strip()


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
        "- 마지막 문장은 감지된 러시아어 표현을 다시 사용해 보라는 학습 안내로 마무리한다.\n"
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
            "이번 문장에서는 등록된 러시아어 학습 키워드가 뚜렷하게 감지되지 않았습니다. "
            "인식된 문장을 확인하고, 필요한 단어나 표현을 다시 말해 보세요."
        )

    details = " ".join(line.strip() for line in rule_feedback.splitlines() if line.strip())
    return (
        "STT 결과에서 학습에 도움이 되는 단서가 확인되었습니다. "
        f"{details} "
        "이 내용을 바탕으로 감지된 러시아어 표현의 의미를 다시 정리해 보세요."
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

def correct_stt_text(
    stt_text: str,
    candidates: Optional[dict] = None,
    scored_candidates: Optional[list[dict]] = None,
) -> str:
    if not stt_text.strip() and not candidates:
        return stt_text

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return stt_text

    candidate_text = ""
    if candidates:
        candidate_text = "\n".join(
            f"- {lang}: {text}"
            for lang, text in candidates.items()
            if text and not str(text).startswith("[ERROR]")
        )
    
    scored_candidate_text = ""

    if scored_candidates:
        scored_candidate_text = "\n".join(
            f"- {item['name']} | score={item['score']} | {item['text']}"
            for item in scored_candidates
            if item.get("text") and not str(item.get("text")).startswith("[ERROR]")
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        user_prompt = f"""
현재 문장은 한국어+러시아어 코드스위칭 학습 발화일 가능성이 높다.

Whisper가 여러 언어 설정으로 인식한 후보는 아래와 같다.

[ASR 후보]
{candidate_text if candidate_text else "- 후보 없음"}

[점수 기반 ASR 후보]
{scored_candidate_text if scored_candidate_text else "- 점수 후보 없음"}

[대표 STT 결과]
{stt_text}

작업:
- 여러 후보를 비교해서 가장 자연스러운 한국어+러시아어 코드스위칭 문장으로 복원하라.
- 한국어 부분은 자연스러운 한국어로 정리한다.
- 러시아어처럼 들리는 부분은 가능한 경우 러시아어 원문으로 복원한다.
- 단, 후보에 전혀 근거가 없는 새 내용을 만들지 않는다.
- 최종 복원 문장 하나만 출력한다.
- 후보에 여러 러시아어 단어가 나열되어 있더라도, 실제 발화로 보이지 않으면 출력하지 마라.
- 단어 목록처럼 보이는 후보는 무시하고, 문장 형태의 후보를 우선 참고하라.
- 한국어로 표기된 러시아어 발음이 있으면 실제 러시아어 단어로 복원하라.
- 예: 슈토는 Что, 야는 Я, 하추는 хочу, 스콜카는 сколько, 파치무는 почему로 복원한다.
- 최종 corrected_text에는 가능하면 복원된 러시아어 원문을 사용한다.
- 한국어 문장 안에서 의미가 어색한 단어는 러시아어 발음 표기일 수 있다고 보고 검토하라.
- 특히 "뜻이 뭐야", "뭐라고 불러", "어떻게 말해", "맞아?" 같은 질문에서는 앞 단어가 러시아어 발음일 가능성이 높다.
- "드롭 밑에 있는 숫자"처럼 학습 질문 문맥에서 해석하면 "дробь 밑에 있는 숫자"일 수 있으므로 러시아어 후보로 복원하라.
- 확실하지 않으면 한국어로 단정하지 말고, 러시아어 후보를 uncertain 토큰으로 남겨라.
- 한국어 음절 사이에 불필요한 조사나 공백이 끼어 있어도 러시아어 발음 후보로 재구성하라.
- 예: "뭐 주요나", "모 주요나", "모쥬나", "모 줘나"는 Можно 후보일 수 있다.
- 사용자가 "한국어로 뭐라고 해", "뜻이 뭐야", "뭐라고 불러"라고 물으면 그 앞의 어색한 단어는 러시아어 발음일 가능성이 높다.

복원 결과:
""".strip()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        corrected = response.choices[0].message.content.strip()
        return corrected or stt_text

    except Exception as e:
        print(f"[LLM STT CORRECTION ERROR] {type(e).__name__}: {e}")
        return stt_text

def analyze_codeswitch_text(stt_text: str, candidates: Optional[dict] = None) -> dict:
    if not stt_text.strip() and not candidates:
        return {
            "corrected_text": stt_text,
            "tokens": [],
            "summary": "분석할 STT 결과가 없습니다.",
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "corrected_text": stt_text,
            "tokens": [],
            "summary": "OPENAI_API_KEY가 없어 LLM 분석을 건너뛰었습니다.",
        }

    candidate_text = ""
    if candidates:
        candidate_text = "\n".join(
            f"- {lang}: {text}"
            for lang, text in candidates.items()
            if text and not str(text).startswith("[ERROR]")
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = f"""
너는 한국어-러시아어 코드스위칭 STT 분석기다.

입력은 Whisper가 인식한 STT 후보들이다.
할 일은 다음과 같다.

1. 가장 자연스러운 한국어+러시아어 문장으로 복원한다.
2. 문장을 한국어/러시아어/불확실 토큰으로 나눈다.
3. 러시아어 토큰은 가능한 경우 한국어 뜻을 적는다.
4. 반드시 JSON만 출력한다.

JSON 형식:
{{
  "corrected_text": "복원된 최종 문장",
  "tokens": [
    {{
      "text": "토큰",
      "language": "ko | ru | uncertain",
      "meaning": "한국어 뜻 또는 빈 문자열",
      "confidence": 0.0
    }}
  ],
  "summary": "짧은 한국어 분석 요약"
}}

규칙:
- JSON 외의 설명을 쓰지 마라.
- confidence는 0.0에서 1.0 사이 숫자다.
- 러시아어처럼 들리지만 확실하지 않으면 language를 uncertain으로 둔다.
- 후보에 전혀 없는 내용을 새로 만들지 마라.
- 한국어로 보이는 단어라도 문맥상 어색하면 러시아어 발음 후보로 분석하라.
- "뜻", "뭐야", "뭐라고 불러", "러시아어로", "한국어로" 같은 표현 주변의 단어는 러시아어 학습 대상일 가능성이 높다.
- 러시아어 후보가 확실하면 language를 "ru"로 두고 키릴 문자로 복원하라.
- 러시아어 후보가 애매하면 language를 "uncertain"으로 두고, corrected_text에는 가장 가능성 높은 후보를 반영하라.
- 분석 결과에는 최소한 하나 이상의 ru 또는 uncertain 토큰을 포함하려고 시도하라. 단, 전혀 근거가 없으면 만들지 않는다.
- 러시아어 발음을 키릴 문자로 단순 음역하지 마라.
- 실제 러시아어 단어로 확신할 수 없으면 language를 uncertain으로 둔다.
- 예: "모쥬나"는 문맥상 "Можно"일 가능성이 높을 때만 "Можно"으로 복원한다.
- "Моу-джу-на"처럼 러시아어 단어가 아닌 음역 표기는 만들지 마라.

[ASR 후보]
{candidate_text if candidate_text else "- 후보 없음"}

[대표 STT]
{stt_text}
""".strip()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "corrected_text": stt_text,
                "tokens": [],
                "summary": "LLM 응답을 JSON으로 해석하지 못했습니다.",
                "raw_llm_output": content,
            }

    except Exception as e:
        return {
            "corrected_text": stt_text,
            "tokens": [],
            "summary": f"LLM 분석 오류: {type(e).__name__}: {e}",
        }


# LLM 피드백 (규칙 기반 중심 → LLM 답변 중심)
def generate_llm_feedback(
    analysis: dict,
    *,
    model: str = "gpt-4o-mini",
) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY가 없어 LLM 피드백을 생성하지 못했습니다."

    corrected_text = analysis.get("corrected_text", "")
    tokens = analysis.get("tokens", [])
    summary = analysis.get("summary", "")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = f"""
너는 한국어-러시아어 코드스위칭 학습 도우미다.

아래 분석 결과를 바탕으로 사용자의 발화에 직접 답변해라.

[복원된 발화]
{corrected_text}

[토큰 분석]
{json.dumps(tokens, ensure_ascii=False, indent=2)}

[분석 요약]
{summary}

답변 규칙:
1. 한국어로 답변한다.
2. 전체는 2~3문장으로 작성한다.
3. 러시아어 표현이 있으면 그 뜻을 설명한다.
4. 사용자가 뜻을 물어보면 뜻을 직접 알려준다.
5. 사용자가 맞는지 물어보면 맞는지/어떻게 고치면 좋은지 알려준다.
6. 가능하면 짧은 예문 하나를 제시한다.
7. "코드스위칭이 감지되었습니다" 같은 기계적인 표현은 필요할 때만 쓴다.
8. 모르면 모른다고 말하고, 인식된 문장을 다시 확인하라고 안내한다.

최종 답변:
""".strip()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "너는 친절하고 간결한 한국어-러시아어 학습 피드백 도우미다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"LLM 피드백 생성 중 오류가 발생했습니다: {type(e).__name__}: {e}"
