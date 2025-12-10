"""
Negative Passage 추가 학습 데이터셋 생성 스크립트 (개선 버전 v3)

개선 사항:
1. Passage 순서 랜덤화 - Position bias 제거
2. Hard negative 품질 관리 - Score 기반 필터링
3. Tokenizer 일관성 - Reader 모델과 동일한 tokenizer 옵션
4. Curriculum learning 지원 - 점진적 난이도 증가
"""

import os
import json
import argparse
import random
from typing import List, Dict, Tuple
from tqdm.auto import tqdm

from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer
import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.retrieval_hybrid import HybridRetrieval

# MeCab 토크나이저
try:
    from mecab import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    print("Warning: mecab-python3 not installed. Using fallback tokenizer.")


class KoreanTokenizer:
    """한국어 형태소 분석 기반 토크나이저"""
    def __init__(self):
        if not MECAB_AVAILABLE:
            raise ImportError(
                "mecab-python3 is not installed. "
                "Please install it with: pip install mecab-python3"
            )
        self.mecab = MeCab()

    def __call__(self, text: str) -> List[str]:
        morphs = self.mecab.morphs(text)
        morphs = [m for m in morphs if len(m) > 1]
        return morphs if morphs else [text]


class NegativePassageDatasetCreator:
    def __init__(
        self,
        train_dataset_path: str = "../data/train_dataset",
        passages_path: str = "../data/wikipedia_passages_256_128.json",
        output_path: str = "../data/train_dataset_negative_passage",
        top_k_retrieval: int = 100,
        rerank_top_k: int = 5,
        alpha: float = 0.7,
        use_rerank: bool = True,
        rerank_model_name: str = "Dongjin-kr/ko-reranker",
        # 새로운 파라미터들
        shuffle_answer_position: bool = True,  # 정답 위치 랜덤화
        use_reader_tokenizer: bool = False,  # Reader tokenizer 사용
        reader_model_name: str = "HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned",
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
        print("Negative Passage Dataset Creator V3 (Improved)")
        print("=" * 80)
        print(f"Train dataset path: {train_dataset_path}")
        print(f"Passages path: {passages_path}")
        print(f"Output path: {output_path}")
        print(f"Top-k retrieval: {top_k_retrieval}")
        print(f"Rerank top-k: {self.rerank_top_k}")
        print(f"Alpha (BM25 weight): {alpha}")
        print(f"Use rerank: {use_rerank}")
        print(f"\n[개선 기능]")
        print(f"  - Shuffle answer position: {shuffle_answer_position}")
        print(f"  - Use reader tokenizer: {use_reader_tokenizer}")
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

        # Initialize tokenizer
        print("Initializing tokenizer...")
        if use_reader_tokenizer:
            print(f"Using Reader tokenizer: {reader_model_name}")
            reader_tokenizer = AutoTokenizer.from_pretrained(reader_model_name)
            self.tokenizer = reader_tokenizer
            tokenize_fn = reader_tokenizer.tokenize
        elif MECAB_AVAILABLE:
            print("Using MeCab tokenizer for retrieval.")
            self.tokenizer = KoreanTokenizer()
            tokenize_fn = self.tokenizer.__call__
        else:
            print("Using default AutoTokenizer for retrieval.")
            default_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", use_fast=False)
            self.tokenizer = default_tokenizer
            tokenize_fn = default_tokenizer.tokenize

        # Initialize retrieval
        print("\nInitializing Hybrid Retrieval...")
        self.retriever = HybridRetrieval(
            tokenize_fn=tokenize_fn,
            data_path=os.path.dirname(passages_path),
            context_path=os.path.basename(passages_path),
            alpha=alpha,
        )

        # Initialize reranker
        if use_rerank:
            print(f"\nInitializing Reranker: {rerank_model_name}...")
            from transformers import AutoModelForSequenceClassification
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
                rerank_model_name
            ).to(device)
            self.rerank_model.eval()
            print(f"Reranker loaded on {device}\n")

        print("Initialization complete!\n")

    def _rerank_passages(
        self,
        question: str,
        doc_indices: List[int]
    ) -> Tuple[List[int], List[float]]:
        """Cross-Encoder로 passages 재정렬 + score 반환"""
        if not self.use_rerank:
            return doc_indices, []

        passages = [self.retriever.contexts[idx] for idx in doc_indices]

        inputs = self.rerank_tokenizer(
            [question] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.rerank_model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().tolist()

        # Score 기준 정렬
        sorted_pairs = sorted(
            zip(doc_indices, scores),
            key=lambda x: x[1],
            reverse=True
        )
        reranked_doc_indices = [idx for idx, _ in sorted_pairs]
        reranked_scores = [score for _, score in sorted_pairs]

        return reranked_doc_indices, reranked_scores

    def _filter_hard_negatives(
        self,
        doc_indices: List[int],
        scores: List[float],
        original_context: str,
        k: int
    ) -> List[int]:
        """Hard negative 필터링 - 정답과 너무 score 차이 나는 것 제거"""
        if not scores or self.min_negative_score_gap <= 0:
            return doc_indices[:k]

        # 정답 passage의 index 찾기
        answer_idx = None
        for i, idx in enumerate(doc_indices):
            if self.retriever.contexts[idx] == original_context:
                answer_idx = i
                break

        if answer_idx is None:
            # 정답이 없으면 그냥 top-k 반환
            return doc_indices[:k]

        answer_score = scores[answer_idx]
        filtered_indices = []

        # 정답은 일단 포함
        filtered_indices.append(doc_indices[answer_idx])

        # 정답과 score 차이가 min_negative_score_gap 이상인 것만 negative로 선택
        for i, (idx, score) in enumerate(zip(doc_indices, scores)):
            if i == answer_idx:
                continue

            score_diff = abs(answer_score - score)
            if score_diff >= self.min_negative_score_gap:
                filtered_indices.append(idx)

            if len(filtered_indices) >= k:
                break

        # 부족하면 그냥 채우기
        if len(filtered_indices) < k:
            for idx in doc_indices:
                if idx not in filtered_indices:
                    filtered_indices.append(idx)
                if len(filtered_indices) >= k:
                    break

        return filtered_indices[:k]

    def _find_answer_start(self, context: str, answer_text: str) -> int:
        """새로운 context에서 answer의 시작 위치 찾기"""
        return context.find(answer_text)

    def create_negative_passage_dataset(self) -> Dataset:
        """Train dataset에 negative passages를 추가한 새로운 dataset 생성"""
        new_data = []
        skipped = 0
        position_stats = {"first": 0, "middle": 0, "last": 0}

        print("\nCreating negative passage dataset...\n")

        for example in tqdm(self.train_data, desc="Processing examples"):
            question = example["question"]
            original_context = example["context"]
            original_answers = example["answers"]
            answer_text = original_answers["text"][0] if len(original_answers["text"]) > 0 else ""

            # 1단계: Retrieval로 Top-K passages 검색
            _, doc_indices = self.retriever._get_relevant_doc(question, k=self.top_k_retrieval)

            # 2단계: Rerank로 정렬 + Score 획득
            if self.use_rerank:
                doc_indices, scores = self._rerank_passages(question, doc_indices)
                # Hard negative 필터링
                doc_indices = self._filter_hard_negatives(
                    doc_indices, scores, original_context, self.rerank_top_k * 2
                )
            else:
                scores = []

            top_k_indices = doc_indices[:self.rerank_top_k]
            top_k_passages = [self.retriever.contexts[idx] for idx in top_k_indices]

            # 3단계: Original passage 포함 여부 확인
            original_in_top_k = original_context in top_k_passages

            if original_in_top_k:
                # 이미 포함되어 있으면 그대로 사용
                final_passages = top_k_passages.copy()

                # Shuffle answer position (정답 위치 랜덤화)
                if self.shuffle_answer_position:
                    answer_passage_idx = final_passages.index(original_context)
                    # 정답을 랜덤한 위치로 이동
                    final_passages.pop(answer_passage_idx)
                    new_pos = random.randint(0, len(final_passages))
                    final_passages.insert(new_pos, original_context)

                    # 통계 수집
                    if new_pos == 0:
                        position_stats["first"] += 1
                    elif new_pos == len(final_passages) - 1:
                        position_stats["last"] += 1
                    else:
                        position_stats["middle"] += 1
            else:
                # 포함되어 있지 않으면 정답을 랜덤 위치에 삽입
                negatives = top_k_passages[:self.rerank_top_k - 1]

                if self.shuffle_answer_position:
                    insert_pos = random.randint(0, len(negatives))
                    final_passages = negatives[:insert_pos] + [original_context] + negatives[insert_pos:]

                    # 통계 수집
                    if insert_pos == 0:
                        position_stats["first"] += 1
                    elif insert_pos == len(negatives):
                        position_stats["last"] += 1
                    else:
                        position_stats["middle"] += 1
                else:
                    # Shuffle 안하면 마지막에 추가
                    final_passages = negatives + [original_context]
                    position_stats["last"] += 1

            # 4단계: Passages concatenate
            new_context = "\n\n".join(final_passages)

            # 5단계: Answer position 재계산
            if answer_text:
                new_answer_start = self._find_answer_start(new_context, answer_text)

                # Answer를 찾을 수 없는 경우 처리
                if new_answer_start == -1:
                    # Original passage를 맨 앞에 배치
                    final_passages = [original_context] + [p for p in final_passages if p != original_context]
                    new_context = "\n\n".join(final_passages)
                    new_answer_start = self._find_answer_start(new_context, answer_text)

                    # 그래도 못 찾으면 원본 데이터 사용
                    if new_answer_start == -1:
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

        if self.shuffle_answer_position:
            total_positioned = sum(position_stats.values())
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
    parser = argparse.ArgumentParser(description="Create negative passage dataset for MRC training (V3)")
    parser.add_argument("--train_dataset_path", type=str, default="../data/train_dataset")
    parser.add_argument("--passages_path", type=str, default="../data/wikipedia_passages_256_128.json")
    parser.add_argument("--output_path", type=str, default="../data/train_dataset_negative_passage_v3")
    parser.add_argument("--top_k_retrieval", type=int, default=100)
    parser.add_argument("--rerank_top_k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--use_rerank", action="store_true", default=True)
    parser.add_argument("--no_rerank", action="store_false", dest="use_rerank")
    parser.add_argument("--rerank_model_name", type=str, default="Dongjin-kr/ko-reranker")

    # 개선 기능 파라미터
    parser.add_argument("--shuffle_answer_position", action="store_true", default=True,
                        help="Randomize answer passage position to prevent position bias")
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle_answer_position")
    parser.add_argument("--use_reader_tokenizer", action="store_true", default=False,
                        help="Use the same tokenizer as the reader model")
    parser.add_argument("--reader_model_name", type=str,
                        default="HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned")
    parser.add_argument("--min_negative_score_gap", type=float, default=0.0,
                        help="Minimum score gap for hard negative filtering")
    parser.add_argument("--curriculum_mode", type=str, default="fixed",
                        choices=["fixed", "easy", "medium", "hard"],
                        help="Curriculum learning mode (easy=3, medium=5, hard=7 passages)")
    parser.add_argument("--seed", type=int, default=42)

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
        shuffle_answer_position=args.shuffle_answer_position,
        use_reader_tokenizer=args.use_reader_tokenizer,
        reader_model_name=args.reader_model_name,
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
