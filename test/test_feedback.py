import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feedback_generator import generate_feedback
from src.llm_postprocessor import postprocess_feedback


SAMPLE_SENTENCES = [
    "오늘은 분수에서 분자와 분모를 배웠습니다.",
    "дробь означает 분수이고 числитель는 분자입니다.",
    "방정식에서 변수와 상수를 구분했습니다.",
    "함수의 그래프에서 기울기와 절편을 확인했습니다.",
    "확률과 평균은 자료를 해석할 때 자주 사용됩니다.",
    "уравнение과 функция라는 러시아어 표현도 함께 들렸습니다.",
    "오늘 수업은 아주 쉬웠어요.",
]


def main():
    for index, sentence in enumerate(SAMPLE_SENTENCES, start=1):
        print(f"[샘플 {index}] {sentence}")
        rule_feedback = generate_feedback(sentence)
        final_feedback = postprocess_feedback(sentence, rule_feedback, use_llm=True)
        print("[규칙 기반 피드백]")
        print(rule_feedback)
        print("[후처리 피드백]")
        print(final_feedback)
        print()


if __name__ == "__main__":
    main()
