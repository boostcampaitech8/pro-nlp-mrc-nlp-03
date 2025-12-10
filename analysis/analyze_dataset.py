"""
생성된 Negative Passage 데이터셋 분석 스크립트

분석 항목:
1. 정답 passage 위치 분포
2. Passage 개수 통계
3. Context 길이 분포
4. Answer position 분포
"""

import argparse
from datasets import load_from_disk
from collections import Counter
import numpy as np


def analyze_dataset(dataset_path: str):
    print("=" * 80)
    print(f"데이터셋 분석: {dataset_path}")
    print("=" * 80)

    # Load dataset
    dataset = load_from_disk(dataset_path)
    train_data = dataset["train"]

    print(f"\n1. 기본 정보")
    print(f"   - Train samples: {len(train_data)}")
    print(f"   - Validation samples: {len(dataset['validation'])}")

    # Context 길이 분석
    context_lengths = []
    passage_counts = []
    answer_positions = []
    answer_char_positions = []

    for example in train_data:
        context = example["context"]
        context_lengths.append(len(context))

        # Passage 개수 (구분자로 계산)
        num_passages = context.count("\n\n") + 1
        passage_counts.append(num_passages)

        # Answer position (몇 번째 passage에 있는지)
        answer_start = example["answers"]["answer_start"][0]
        answer_char_positions.append(answer_start)

        # Answer가 몇 번째 passage에 있는지 계산
        passages = context.split("\n\n")
        current_pos = 0
        answer_passage_idx = -1

        for i, passage in enumerate(passages):
            passage_end = current_pos + len(passage)
            if current_pos <= answer_start < passage_end:
                answer_passage_idx = i
                break
            current_pos = passage_end + 2  # "\n\n" 길이

        answer_positions.append(answer_passage_idx)

    # 통계 출력
    print(f"\n2. Context 길이 통계")
    print(f"   - 평균: {np.mean(context_lengths):.1f} 자")
    print(f"   - 중앙값: {np.median(context_lengths):.1f} 자")
    print(f"   - 최소: {min(context_lengths)} 자")
    print(f"   - 최대: {max(context_lengths)} 자")

    print(f"\n3. Passage 개수 분포")
    passage_count_dist = Counter(passage_counts)
    for count in sorted(passage_count_dist.keys()):
        percentage = passage_count_dist[count] / len(train_data) * 100
        print(f"   - {count} passages: {passage_count_dist[count]} 개 ({percentage:.1f}%)")

    print(f"\n4. 정답 Passage 위치 분포")
    answer_position_dist = Counter(answer_positions)
    for pos in sorted(answer_position_dist.keys()):
        percentage = answer_position_dist[pos] / len(train_data) * 100
        position_name = "첫 번째" if pos == 0 else f"{pos+1}번째"
        print(f"   - {position_name} passage: {answer_position_dist[pos]} 개 ({percentage:.1f}%)")

    # Position bias 확인
    if len(passage_count_dist) == 1:
        num_passages = list(passage_count_dist.keys())[0]
        first_pos_count = answer_position_dist.get(0, 0)
        last_pos_count = answer_position_dist.get(num_passages - 1, 0)
        middle_pos_count = sum(
            answer_position_dist.get(i, 0)
            for i in range(1, num_passages - 1)
        )

        print(f"\n5. Position Bias 분석 ({num_passages} passages 기준)")
        print(f"   - 첫 번째 위치: {first_pos_count} 개 ({first_pos_count/len(train_data)*100:.1f}%)")
        print(f"   - 중간 위치: {middle_pos_count} 개 ({middle_pos_count/len(train_data)*100:.1f}%)")
        print(f"   - 마지막 위치: {last_pos_count} 개 ({last_pos_count/len(train_data)*100:.1f}%)")

        # Position bias 평가
        expected_ratio = 100 / num_passages
        if abs(first_pos_count/len(train_data)*100 - expected_ratio) < 5:
            print(f"   ✅ Position bias가 적절히 완화되었습니다!")
        else:
            print(f"   ⚠️  Position bias가 존재할 수 있습니다.")

    # 샘플 출력
    print(f"\n6. 샘플 예시")
    print("-" * 80)
    sample = train_data[0]
    print(f"Question: {sample['question']}")
    print(f"Context (first 300 chars):\n{sample['context'][:300]}...")
    print(f"Answer: {sample['answers']['text']}")
    print(f"Answer start: {sample['answers']['answer_start'][0]}")
    print("-" * 80)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze negative passage dataset")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    args = parser.parse_args()

    analyze_dataset(args.dataset_path)
