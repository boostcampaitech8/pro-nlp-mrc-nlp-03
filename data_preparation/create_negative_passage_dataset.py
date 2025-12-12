"""
Negative Passage 추가 학습 데이터셋 생성 스크립트 (개선 버전 v4)

개선 사항:
1. RRF (Reciprocal Rank Fusion) 기반 Hybrid Retrieval 사용
2. Cross-Encoder Rerank 적용 (inference와 동일한 로직)
3. Passage 순서 랜덤화 - Position bias 제거
4. Curriculum learning 지원 - 점진적 난이도 증가 (easy=3, medium=5, hard=7)
5. Batch 단위 Rerank 처리로 메모리 효율성 향상
"""

import os
import json
import argparse
import random
from tqdm.auto import tqdm

from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.retrieval_hybrid_passage_rerank_only import HybridRetrievalRerankRRF


class NegativePassageDatasetCreator:
    def __init__(
        self,
        train_dataset_path: str = "../../data/train_dataset",
        passages_path: str = "../../data/wikipedia_passages_256_128.json",
        output_path: str = "../../data/train_dataset_negative_passage",
        top_k_retrieval: int = 100,
        rerank_top_k: int = 5,
        alpha: float = 0.7,
        use_rerank: bool = True,
        rerank_model_name: str = "Dongjin-kr/ko-reranker",
        rerank_candidate_k: int = 20,  # RRF Hybrid에서 먼저 뽑을 개수
        rerank_batch_size: int = 32,  # Rerank 배치 사이즈
        # 새로운 파라미터들
        shuffle_answer_position: bool = True,  # 정답 위치 랜덤화
        min_negative_score_gap: float = 0.0,  # Hard negative 최소 score 차이
        curriculum_mode: str = "fixed",  # "fixed", "easy", "medium", "hard"
        seed: int = 42,
    ):
        self.train_dataset_path = train_dataset_path
        self.passages_path = passages_path
        self.output_path = output_path
        self.top_k_retrieval = top_k_retrieval
        self.alpha = alpha
        self.use_rerank = use_rerank
        self.rerank_model_name = rerank_model_name
        self.rerank_candidate_k = rerank_candidate_k
        self.rerank_batch_size = rerank_batch_size
        self.shuffle_answer_position = shuffle_answer_position
        self.min_negative_score_gap = min_negative_score_gap
        self.seed = seed

        # Curriculum learning 설정
        self.curriculum_mode = curriculum_mode
        if curriculum_mode == "easy":
            self.rerank_top_k = 3
        elif curriculum_mode == "medium":
            self.rerank_top_k = 5
        elif curriculum_mode == "hard":
            self.rerank_top_k = 7
        else:
            self.rerank_top_k = rerank_top_k

        random.seed(seed)

        print("=" * 80)
        print("Negative Passage Dataset Creator V4 (RRF + Rerank)")
        print("=" * 80)
        print(f"Train dataset path: {train_dataset_path}")
        print(f"Passages path: {passages_path}")
        print(f"Output path: {output_path}")
        print(f"Top-k retrieval: {top_k_retrieval}")
        print(f"Rerank candidate k: {rerank_candidate_k}")
        print(f"Rerank top-k: {self.rerank_top_k}")
        print(f"Rerank batch size: {rerank_batch_size}")
        print(f"Alpha (BM25 weight): {alpha}")
        print(f"Use rerank: {use_rerank}")
        print(f"\n[개선 기능]")
        print(f"  - Shuffle answer position: {shuffle_answer_position}")
        print(f"  - Min negative score gap: {min_negative_score_gap}")
        print(f"  - Curriculum mode: {curriculum_mode}")
        print(f"  - Random seed: {seed}")
        print("=" * 80 + "\n")

        # Load dataset
        print("Loading train dataset...")
        self.dataset = load_from_disk(train_dataset_path)
        self.train_data = self.dataset["train"]
        print(f"Train dataset size: {len(self.train_data)}\n")

        # Load passages
        print("Loading Wikipedia passages...")
        with open(passages_path, "r", encoding="utf-8") as f:
            passages_dict = json.load(f)
        self.passages = passages_dict
        print(f"Total passages: {len(passages_dict)}\n")

        # Initialize tokenizer (using default AutoTokenizer like inference)
        print("Initializing tokenizer...")
        print("Using default AutoTokenizer for retrieval (klue/bert-base).")
        default_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", use_fast=False)
        self.tokenizer = default_tokenizer
        tokenize_fn = default_tokenizer.tokenize

        # Initialize retrieval (RRF + Rerank like inference)
        print("\nInitializing Hybrid Retrieval (RRF + Rerank)...")
        self.retriever = HybridRetrievalRerankRRF(
            tokenize_fn=tokenize_fn,
            data_path=os.path.dirname(passages_path),
            context_path=os.path.basename(passages_path),
            alpha=alpha,
            use_rerank=use_rerank,
            rerank_model_name=rerank_model_name,
            rerank_candidate_k=rerank_candidate_k,
            rerank_batch_size=rerank_batch_size,
        )

        print("Initialization complete!\n")

    def _find_answer_start(self, context: str, answer_text: str) -> int:
        """새로운 context에서 answer의 시작 위치 찾기"""
        return context.find(answer_text)

    def _extract_answer_passage(self, context: str, answer_text: str, max_length: int = 512) -> str:
        """
        원본 context에서 정답 주변의 passage를 추출

        Args:
            context: 원본 context (긴 문서)
            answer_text: 정답 텍스트
            max_length: 추출할 최대 길이 (문자 단위)

        Returns:
            정답을 포함하는 passage
        """
        if not answer_text or answer_text not in context:
            # 정답이 없으면 앞부분만 반환
            return context[:max_length]

        answer_start = context.find(answer_text)
        answer_end = answer_start + len(answer_text)

        # 정답을 중심으로 앞뒤로 context 추출
        # 정답이 중앙에 오도록 조정
        half_length = max_length // 2

        start = max(0, answer_start - half_length)
        end = min(len(context), answer_end + half_length)

        # 실제 길이가 max_length보다 짧으면 조정
        if end - start < max_length:
            if start == 0:
                end = min(len(context), start + max_length)
            elif end == len(context):
                start = max(0, end - max_length)

        passage = context[start:end]

        # 문장 경계에서 자르기 (선택적)
        # 앞쪽 정리
        if start > 0:
            # 첫 문장이 잘렸을 수 있으므로 첫 번째 마침표 이후부터 시작
            first_period = passage.find('. ')
            if first_period != -1 and first_period < len(passage) // 4:
                passage = passage[first_period + 2:]

        # 뒤쪽 정리
        if end < len(context):
            # 마지막 문장이 잘렸을 수 있으므로 마지막 마침표까지만 포함
            last_period = passage.rfind('. ')
            if last_period != -1 and last_period > len(passage) * 3 // 4:
                passage = passage[:last_period + 1]

        return passage.strip()

    def create_negative_passage_dataset(self) -> Dataset:
        """Train dataset에 negative passages를 추가한 새로운 dataset 생성"""
        new_data = []
        skipped = 0
        position_stats = {"first": 0, "middle": 0, "last": 0}
        answer_found_stats = {"found": 0, "not_found": 0}

        print("\nCreating negative passage dataset...\n")

        for example in tqdm(self.train_data, desc="Processing examples"):
            question = example["question"]
            original_context = example["context"]
            original_answers = example["answers"]
            answer_text = original_answers["text"][0] if len(original_answers["text"]) > 0 else ""

            # 1단계: RRF Hybrid Retrieval + Rerank로 Top-K passages 검색
            # HybridRetrievalRerankRRF는 내부에서 RRF 점수로 상위 rerank_candidate_k개를 뽑고,
            # Cross-Encoder로 재정렬한 뒤 최종 top-k를 반환합니다.
            _, doc_indices = self.retriever._get_relevant_doc(question, k=self.rerank_top_k)

            top_k_indices = doc_indices[:self.rerank_top_k]
            top_k_passages = [self.retriever.contexts[idx] for idx in top_k_indices]

            # 2단계: 정답이 포함된 passage 찾기
            # answer_text가 실제로 어느 passage에 포함되어 있는지 확인
            answer_passage_idx = None
            if answer_text:
                for idx, passage in enumerate(top_k_passages):
                    if answer_text in passage:
                        answer_passage_idx = idx
                        break

            # 3단계: Final passages 구성
            if answer_passage_idx is not None:
                # 정답이 포함된 passage가 검색된 경우
                answer_found_stats["found"] += 1
                final_passages = top_k_passages.copy()

                # Shuffle answer position (정답 위치 랜덤화)
                if self.shuffle_answer_position:
                    # 정답 passage를 제거하고 랜덤한 위치에 다시 삽입
                    answer_passage = final_passages.pop(answer_passage_idx)
                    new_pos = random.randint(0, len(final_passages))
                    final_passages.insert(new_pos, answer_passage)

                    # 통계 수집
                    if new_pos == 0:
                        position_stats["first"] += 1
                    elif new_pos == len(final_passages) - 1:
                        position_stats["last"] += 1
                    else:
                        position_stats["middle"] += 1
                else:
                    # Shuffle 안하면 원래 위치 유지
                    # 통계 수집
                    if answer_passage_idx == 0:
                        position_stats["first"] += 1
                    elif answer_passage_idx == len(final_passages) - 1:
                        position_stats["last"] += 1
                    else:
                        position_stats["middle"] += 1
            else:
                # 정답이 포함된 passage가 없는 경우
                # 검색된 passage들을 negative로 사용하고, 정답이 포함된 passage를 별도로 추가
                answer_found_stats["not_found"] += 1

                # 정답이 포함된 passage 찾기 (원본 context에서)
                # 원본 context를 그대로 사용하는 대신, 정답 주변만 추출
                answer_passage = self._extract_answer_passage(original_context, answer_text)

                # rerank_top_k - 1개의 negative passages 사용
                negatives = top_k_passages[:self.rerank_top_k - 1]

                if self.shuffle_answer_position:
                    insert_pos = random.randint(0, len(negatives))
                    final_passages = negatives[:insert_pos] + [answer_passage] + negatives[insert_pos:]

                    # 통계 수집
                    if insert_pos == 0:
                        position_stats["first"] += 1
                    elif insert_pos == len(negatives):
                        position_stats["last"] += 1
                    else:
                        position_stats["middle"] += 1
                else:
                    # Shuffle 안하면 마지막에 추가
                    final_passages = negatives + [answer_passage]
                    position_stats["last"] += 1

            # 4단계: Passages concatenate
            new_context = "\n\n".join(final_passages)

            # 5단계: Answer position 재계산
            if answer_text:
                new_answer_start = self._find_answer_start(new_context, answer_text)

                # Answer를 찾을 수 없는 경우 처리 (거의 발생하지 않아야 함)
                if new_answer_start == -1:
                    # 이미 _extract_answer_passage()로 정답 주변을 추출했는데도 못 찾는 경우
                    # 원본 context 전체를 사용
                    skipped += 1
                    new_data.append({
                        "id": example["id"],
                        "question": question,
                        "context": original_context,
                        "answers": original_answers,
                        "document_id": example.get("document_id", ""),
                        "__index_level_0__": example.get("__index_level_0__", 0),
                    })
                    continue
            else:
                new_answer_start = 0

            # 6단계: 새로운 example 생성
            new_example = {
                "id": example["id"],
                "question": question,
                "context": new_context,
                "answers": {
                    "text": original_answers["text"],
                    "answer_start": [new_answer_start],
                },
                "document_id": example.get("document_id", ""),
                "__index_level_0__": example.get("__index_level_0__", 0),
            }

            new_data.append(new_example)

        print(f"\nDataset creation complete!")
        print(f"Total examples: {len(new_data)}")
        print(f"Skipped (answer not found): {skipped}")
        print(f"Success rate: {(len(new_data) - skipped) / len(new_data) * 100:.2f}%")

        # Retrieval 성공률 출력
        total_retrieval = answer_found_stats["found"] + answer_found_stats["not_found"]
        if total_retrieval > 0:
            print(f"\nRetrieval Statistics:")
            print(f"  - Answer found in retrieved passages: {answer_found_stats['found']} ({answer_found_stats['found']/total_retrieval*100:.1f}%)")
            print(f"  - Answer NOT found (used original context): {answer_found_stats['not_found']} ({answer_found_stats['not_found']/total_retrieval*100:.1f}%)")

        if self.shuffle_answer_position:
            total_positioned = sum(position_stats.values())
            if total_positioned > 0:
                print(f"\nAnswer Position Distribution:")
                print(f"  - First position: {position_stats['first']} ({position_stats['first']/total_positioned*100:.1f}%)")
                print(f"  - Middle position: {position_stats['middle']} ({position_stats['middle']/total_positioned*100:.1f}%)")
                print(f"  - Last position: {position_stats['last']} ({position_stats['last']/total_positioned*100:.1f}%)")

        return Dataset.from_dict({
            "id": [d["id"] for d in new_data],
            "question": [d["question"] for d in new_data],
            "context": [d["context"] for d in new_data],
            "answers": [d["answers"] for d in new_data],
            "document_id": [d["document_id"] for d in new_data],
            "__index_level_0__": [d["__index_level_0__"] for d in new_data],
        })

    def save_dataset(self, new_train_dataset: Dataset):
        """새로운 dataset 저장"""
        new_dataset_dict = DatasetDict({
            "train": new_train_dataset,
            "validation": self.dataset["validation"]
        })

        print(f"\nSaving dataset to {self.output_path}...")
        new_dataset_dict.save_to_disk(self.output_path)
        print(f"Dataset saved successfully!\n")

        # 샘플 출력
        print("=" * 80)
        print("Sample Example (First)")
        print("=" * 80)
        sample = new_train_dataset[0]
        print(f"ID: {sample['id']}")
        print(f"Question: {sample['question']}")
        print(f"Context (first 500 chars):\n{sample['context'][:500]}...")
        print(f"\nAnswer: {sample['answers']['text']}")
        print(f"Answer start: {sample['answers']['answer_start']}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Create negative passage dataset for MRC training (V4 - RRF + Rerank)")
    parser.add_argument("--train_dataset_path", type=str, default="../data/train_dataset")
    parser.add_argument("--passages_path", type=str, default="../data/wikipedia_passages_256_128.json")
    parser.add_argument("--output_path", type=str, default="../data/train_dataset_negative_passage_v4")
    parser.add_argument("--top_k_retrieval", type=int, default=100,
                        help="Maximum retrieval passages (not used in RRF mode)")
    parser.add_argument("--rerank_top_k", type=int, default=5,
                        help="Final number of passages to include")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="BM25 weight for RRF hybrid retrieval")
    parser.add_argument("--use_rerank", action="store_true", default=True,
                        help="Use cross-encoder reranking")
    parser.add_argument("--no_rerank", action="store_false", dest="use_rerank")
    parser.add_argument("--rerank_model_name", type=str, default="Dongjin-kr/ko-reranker",
                        help="Cross-encoder model for reranking")
    parser.add_argument("--rerank_candidate_k", type=int, default=20,
                        help="Number of candidates to retrieve before reranking")
    parser.add_argument("--rerank_batch_size", type=int, default=32,
                        help="Batch size for reranking")

    # 개선 기능 파라미터
    parser.add_argument("--shuffle_answer_position", action="store_true", default=True,
                        help="Randomize answer passage position to prevent position bias")
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle_answer_position")
    parser.add_argument("--min_negative_score_gap", type=float, default=0.0,
                        help="Minimum score gap for hard negative filtering (deprecated)")
    parser.add_argument("--curriculum_mode", type=str, default="fixed",
                        choices=["fixed", "easy", "medium", "hard"],
                        help="Curriculum learning mode (easy=3, medium=5, hard=7 passages)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    # Create dataset creator
    creator = NegativePassageDatasetCreator(
        train_dataset_path=args.train_dataset_path,
        passages_path=args.passages_path,
        output_path=args.output_path,
        top_k_retrieval=args.top_k_retrieval,
        rerank_top_k=args.rerank_top_k,
        alpha=args.alpha,
        use_rerank=args.use_rerank,
        rerank_model_name=args.rerank_model_name,
        rerank_candidate_k=args.rerank_candidate_k,
        rerank_batch_size=args.rerank_batch_size,
        shuffle_answer_position=args.shuffle_answer_position,
        min_negative_score_gap=args.min_negative_score_gap,
        curriculum_mode=args.curriculum_mode,
        seed=args.seed,
    )

    # Create dataset
    new_train_dataset = creator.create_negative_passage_dataset()

    # Save dataset
    creator.save_dataset(new_train_dataset)

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
