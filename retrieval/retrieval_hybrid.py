# retrieval_hybrid.py (수정)
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from typing import List, Tuple, Union

from .retrieval_bm25 import BM25Retrieval
from .retrieval_dense import DenseRetrieval


class HybridRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: str = "../data",
        # context_path: str = "wikipedia_documents.json",
        context_path="wikipedia_passages_256_128.json",
        # dense_model: str = "jhgan/ko-sroberta-multitask",
        # dense_model="intfloat/multilingual-e5-base",
        dense_model="BAAI/bge-m3",
        alpha: float = 0.7,
    ):
        """
        Sparse(BM25) + Dense 하이브리드 검색

        Arguments:
            alpha: BM25 가중치 (0~1)
        """
        print("[HybridRetrieval] Initializing BM25...")
        self.bm25 = BM25Retrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path,
        )
        self.bm25.get_sparse_embedding()

        print("[HybridRetrieval] Initializing Dense...")
        self.dense = DenseRetrieval(
            data_path=data_path,
            context_path=context_path,
            model_name=dense_model,
        )
        self.dense.build_dense_embedding()

        self.alpha = alpha
        self.contexts = self.bm25.contexts

        assert len(self.bm25.contexts) == len(self.dense.contexts), "Context 개수 불일치!"

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max 정규화 (0~1)"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-9:
            return np.ones_like(scores) * 0.5
        return (scores - min_score) / (max_score - min_score)

    def _get_relevant_doc(self, query: str, k: int) -> Tuple[List[float], List[int]]:
        """
        단일 query에 대한 하이브리드 검색 (수정!)
        """
        # BM25 전체 문서 점수 (인덱스 순서대로!)
        tokenized_query = self.bm25.tokenize_fn(query)
        bm25_all_scores = self.bm25.bm25.get_scores(tokenized_query)  # ← 핵심!

        # Dense 전체 문서 점수 (인덱스 순서대로!)
        q_emb = self.dense._encode_query(query)
        dense_all_scores = np.dot(self.dense.passage_embs, q_emb.T).squeeze(-1)  # ← 핵심!

        # 정규화
        bm25_norm = self._normalize_scores(bm25_all_scores)
        dense_norm = self._normalize_scores(dense_all_scores)

        # 가중 평균
        hybrid_scores = self.alpha * bm25_norm + (1 - self.alpha) * dense_norm

        # Top-k 선택
        top_indices = np.argsort(hybrid_scores)[::-1][:k]
        top_scores = hybrid_scores[top_indices].tolist()
        top_indices = top_indices.tolist()

        return top_scores, top_indices

    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: int = 5,
    ) -> Union[Tuple[List[float], List[str]], pd.DataFrame]:

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self._get_relevant_doc(query_or_dataset, topk)

            print(f"\n[Query] {query_or_dataset}\n")
            print(f"Hybrid (α={self.alpha}, BM25 {self.alpha*100:.0f}% + Dense {(1-self.alpha)*100:.0f}%)\n")

            for i in range(topk):
                print(f"Top-{i+1} (score: {doc_scores[i]:.4f})")
                print(f"{self.contexts[doc_indices[i]][:200]}...\n")

            return doc_scores, [self.contexts[idx] for idx in doc_indices]

        elif isinstance(query_or_dataset, Dataset):
            return self._retrieve_bulk(query_or_dataset, topk)

    def _retrieve_bulk(self, dataset: Dataset, topk: int) -> pd.DataFrame:
        total = []

        for example in tqdm(dataset, desc="Hybrid Retrieval"):
            question = example["question"]
            _, doc_indices = self._get_relevant_doc(question, topk)

            retrieved_contexts = [self.contexts[idx] for idx in doc_indices]

            tmp = {
                "question": question,
                "id": example["id"],
                "context": " ".join(retrieved_contexts),
            }

            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]

            total.append(tmp)

        return pd.DataFrame(total)


if __name__ == "__main__":
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    dataset = load_from_disk("../data/train_dataset")
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", use_fast=False)

    # 단일 query 먼저 테스트
    print("\n" + "="*60)
    print("단일 Query 테스트")
    print("="*60)

    retriever = HybridRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path="../data",
        alpha=0.7,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    scores, contexts = retriever.retrieve(query, topk=3)

    # 여러 alpha 값 테스트
    alphas = [0.5, 0.6, 0.7, 0.8, 0.9]

    print("\n" + "="*60)
    print("Alpha 튜닝 (Validation Set)")
    print("="*60)

    for alpha in alphas:
        print(f"\n[α={alpha}] Testing...")

        retriever = HybridRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="../data",
            alpha=alpha,
        )

        # Validation 테스트
        val_df = retriever.retrieve(dataset["validation"], topk=5)

        if "original_context" in val_df.columns:
            val_df["hit"] = val_df.apply(
                lambda row: row["original_context"] in row["context"], axis=1
            )
            accuracy = val_df["hit"].sum() / len(val_df)
            improvement = (accuracy - 0.8417) * 100

            print(f"✅ Hybrid (α={alpha}): {accuracy:.4f} ({improvement:+.2f}%p)")

    print("\n" + "="*60)
    print("기준선")
    print("="*60)
    print("BM25만: 84.17%")
    print("Dense만: 47.08%")