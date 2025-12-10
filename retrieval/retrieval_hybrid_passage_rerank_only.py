# retrieval_hybrid_passage_rerank_only.py

import numpy as np
from typing import List, Tuple, Union

from datasets import Dataset
import pandas as pd
from tqdm.auto import tqdm

from .retrieval_hybrid_passage import HybridRetrieval as BaseHybridRetrieval

import torch
from transformers import (
    AutoTokenizer as HfAutoTokenizer,
    AutoModelForSequenceClassification,
)


class HybridRetrievalRerank(BaseHybridRetrieval):
    """
    기존 HybridRetrieval (BM25 + Dense + alpha)의 점수로 상위 k_candidate를 뽑고,
    Cross-Encoder reranker로 다시 정렬한 뒤 최종 top-k를 반환하는 리트리버.
    """

    def __init__(
        self,
        *args,
        use_rerank: bool = True,
        rerank_model_name: str = "Dongjin-kr/ko-reranker",
        rerank_candidate_k: int = 20,   # Hybrid에서 먼저 뽑을 개수
        device: str = None,
        **kwargs,
    ):
        # 🔹 alpha, bm25/dense 등 기존 Hybrid 설정은 전부 BaseHybridRetrieval에서 처리
        super().__init__(*args, **kwargs)

        self.use_rerank = use_rerank
        self.rerank_candidate_k = rerank_candidate_k

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.use_rerank:
            print(f"[HybridRetrievalRerank] Loading reranker: {rerank_model_name}")
            self.rerank_tokenizer = HfAutoTokenizer.from_pretrained(rerank_model_name)
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
                rerank_model_name
            ).to(self.device)
            self.rerank_model.eval()

    # -----------------------------
    # Cross-Encoder Rerank
    # -----------------------------
    def _rerank_passages(self, question: str, doc_indices: List[int]) -> List[int]:
        """
        question + 각 passage를 Cross-Encoder에 넣어 relevance score를 얻고
        그 점수로 doc_indices를 재정렬.
        """
        passages = [self.contexts[idx] for idx in doc_indices]

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
            scores = outputs.logits.squeeze(-1)  # (num_passages,)

        sorted_idx = torch.argsort(scores, descending=True).tolist()
        reranked_doc_indices = [doc_indices[i] for i in sorted_idx]
        return reranked_doc_indices

    # -----------------------------
    # bulk 모드에서 rerank 적용
    # -----------------------------
    def _retrieve_bulk(self, dataset: Dataset, topk: int) -> pd.DataFrame:
        """
        dataset: HF Dataset (validation 등)
        topk: 최종적으로 Reader에 넘길 passage 개수 (ex. 5)
        """
        total = []

        for example in tqdm(dataset, desc="Hybrid Retrieval (alpha) + Rerank"):
            question = example["question"]

            # 1단계: 기존 HybridRetrieval 로 상위 rerank_candidate_k 개 뽑기
            k_candidate = self.rerank_candidate_k if self.use_rerank else topk
            _, doc_indices = self._get_relevant_doc(question, k=k_candidate)
            # ↑ 여기서 이미 BM25 0.7 + Dense 0.3 (alpha) 반영된 점수

            # 2단계: Cross-Encoder로 재정렬
            if self.use_rerank:
                doc_indices = self._rerank_passages(question, doc_indices)

            # 3단계: 최종 top-k만 사용
            doc_indices = doc_indices[:topk]
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
