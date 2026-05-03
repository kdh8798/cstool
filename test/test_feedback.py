import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feedback_generator import generate_feedback


SAMPLE_SENTENCES = [
    "오늘은 분수에서 분자와 분모를 배웠습니다.",
    "дробь означает 분수이고 числитель는 분자입니다.",
    "знаменатель는 분수에서 아래에 있는 수입니다.",
    "오늘 수업은 아주 쉬웠어요.",
]


def main():
    for index, sentence in enumerate(SAMPLE_SENTENCES, start=1):
        print(f"[샘플 {index}] {sentence}")
        print(generate_feedback(sentence))
        print()


if __name__ == "__main__":
    main()
