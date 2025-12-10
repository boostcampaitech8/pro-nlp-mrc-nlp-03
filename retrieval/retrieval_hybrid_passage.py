#retrieval_hybrid_passage.py 

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from typing import List, Tuple, Union, Dict

# retrieval_bm25와 retrieval_dense는 이미 import 되었다고 가정합니다.
from retrieval.retrieval_bm25 import BM25Retrieval
from retrieval.retrieval_dense import DenseRetrievalEnsemble

# RRF 상수 K: 일반적으로 60을 사용하며, 순위 융합의 민감도를 조정합니다.
RRF_K = 60 

# =========================================================================
# 1. HybridRetrieval (기존: Alpha-가중치 방식)
# =========================================================================

class HybridRetrieval:
    # (기존 HybridRetrieval 코드는 유지 - 생략)
    # ... (생략) ...
    # __init__, _normalize_scores, _get_relevant_doc (alpha 방식), retrieve, _retrieve_bulk 메서드는 원본과 동일하게 유지됩니다.
    
    def __init__(
        self,
        tokenize_fn,
        data_path: str = "../data",
        context_path: str = "wikipedia_passages_256_128.json",
        dense_model: List[str] = ["jhgan/ko-sroberta-multitask", "BAAI/bge-m3"],
        alpha: float = 0.7,
    ):
        print("[HybridRetrieval] Initializing BM25...")
        self.bm25 = BM25Retrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path,
        )
        self.bm25.get_sparse_embedding()

        print("[HybridRetrieval] Initializing Dense...")
        self.dense = DenseRetrievalEnsemble(
            data_path=data_path,
            context_path=context_path,
            model_names=dense_model, 
        )
        self.dense.build_dense_embedding()
        
        self.alpha = alpha
        self.contexts = self.bm25.contexts  # passage 리스트

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
        단일 query에 대한 하이브리드 검색 (BM25 + Dense, alpha-가중치 방식)
        """
        # 1) BM25 전체 passage 점수 (인덱스 순서대로)
        tokenized_query = self.bm25.tokenize_fn(query)
        bm25_all_scores = self.bm25.bm25.get_scores(tokenized_query)

        # 2) Dense 전체 passage 점수 (인덱스 순서대로)
        q_emb = self.dense._encode_query(query)
        dense_all_scores = np.dot(self.dense.passage_embs, q_emb.T).squeeze(-1)

        # 3) 정규화
        bm25_norm = self._normalize_scores(bm25_all_scores)
        dense_norm = self._normalize_scores(dense_all_scores)

        # 4) 가중 평균 결합
        hybrid_scores = self.alpha * bm25_norm + (1 - self.alpha) * dense_norm

        # 5) Top-k 선택
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
            print(
                f"Hybrid (α={self.alpha}, "
                f"BM25 {self.alpha*100:.0f}% + Dense {(1-self.alpha)*100:.0f}%)\n"
            )

            for i in range(topk):
                print(f"Top-{i+1} (score: {doc_scores[i]:.4f})")
                print(f"{self.contexts[doc_indices[i]][:200]}...\n")

            return doc_scores, [self.contexts[idx] for idx in doc_indices]

        elif isinstance(query_or_dataset, Dataset):
            return self._retrieve_bulk(query_or_dataset, topk)

        else:
            raise TypeError("query_or_dataset는 str 또는 Dataset 이어야 합니다.")

    def _retrieve_bulk(self, dataset: Dataset, topk: int) -> pd.DataFrame:
        """
        Dataset(question들)에 대해 일괄 하이브리드 검색
        """
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

# =========================================================================
# 2. HybridRetrievalRRF (신규: RRF 순위 융합 방식)
# =========================================================================

class HybridRetrievalRRF(HybridRetrieval):
    """
    Reciprocal Rank Fusion (RRF)을 사용하여 BM25와 Concatenation Dense Retrieval 결과를 결합합니다.
    (alpha 대신 RRF_K를 사용하며, RRF_K는 클래스 밖의 상수를 따릅니다.)
    """

    def __init__(self, *args, **kwargs):
        # 기존 HybridRetrieval의 초기화 로직을 그대로 사용 (BM25, Dense 모델 로드)
        super().__init__(*args, **kwargs)

    def _get_relevant_doc(self, query: str, k: int) -> Tuple[List[float], List[int]]:
        """
        단일 query에 대한 하이브리드 검색 (BM25 + Dense, RRF 방식)
        """
        # 1) BM25 전체 passage 점수 및 순위 (rank) 추출
        tokenized_query = self.bm25.tokenize_fn(query)
        bm25_all_scores = self.bm25.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_all_scores)[::-1]
        
        # 2) Dense 전체 passage 점수 및 순위 (rank) 추출
        q_emb = self.dense._encode_query(query)
        dense_all_scores = np.dot(self.dense.passage_embs, q_emb.T).squeeze(-1)
        dense_indices = np.argsort(dense_all_scores)[::-1]
        
        # 3) RRF 점수 계산을 위한 순위 맵 생성
        # Ranks dictionary: {doc_id: RRF_Score}
        # 순위는 1부터 시작합니다. (rank = index + 1)
        rrf_scores: Dict[int, float] = {}

        # BM25 순위 기여
        for rank, doc_id in enumerate(bm25_indices):
            # 순위는 1부터 시작
            r = rank + 1
            score_contribution = 1.0 / (RRF_K + r)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score_contribution

        # Dense 순위 기여
        for rank, doc_id in enumerate(dense_indices):
            r = rank + 1
            score_contribution = 1.0 / (RRF_K + r)
            # 이미 BM25에서 점수가 계산되었다면 더하고, 아니면 0.0을 기본값으로 사용
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score_contribution

        # 4) RRF 점수를 기반으로 Top-k 선택
        # 딕셔너리의 (키, 값)을 (doc_id, score) 형태로 가져와 score 기준으로 내림차순 정렬
        sorted_rrf = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Top-k 인덱스와 점수 추출
        top_indices = [item[0] for item in sorted_rrf][:k]
        top_scores = [item[1] for item in sorted_rrf][:k]

        return top_scores, top_indices

    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: int = 5,
    ) -> Union[Tuple[List[float], List[str]], pd.DataFrame]:
        
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self._get_relevant_doc(query_or_dataset, topk)

            print(f"\n[Query] {query_or_dataset}\n")
            print(f"Hybrid (RRF Fusion, K={RRF_K})\n")

            for i in range(topk):
                print(f"Top-{i+1} (score: {doc_scores[i]:.4f})")
                print(f"{self.contexts[doc_indices[i]][:200]}...\n")

            return doc_scores, [self.contexts[idx] for idx in doc_indices]

        # Dataset 일괄 처리 (_retrieve_bulk)는 부모 클래스의 것을 그대로 사용합니다.
        elif isinstance(query_or_dataset, Dataset):
            return self._retrieve_bulk(query_or_dataset, topk)
        
        else:
            raise TypeError("query_or_dataset는 str 또는 Dataset 이어야 합니다.")


# =========================================================================
# 3. 테스트 코드 (RRF 테스트 추가)
# =========================================================================

if __name__ == "__main__":
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    dataset = load_from_disk("../data/train_dataset")
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", use_fast=False)
    ENSEMBLE_MODEL_NAMES = ["jhgan/ko-sroberta-multitask", "BAAI/bge-m3"]
    
    # ============================
    # RRF Query 테스트
    # ============================
    print("\n" + "=" * 60)
    print("단일 Query 테스트 (Hybrid RRF)")
    print("=" * 60)

    retriever_rrf = HybridRetrievalRRF(
        tokenize_fn=tokenizer.tokenize,
        data_path="../data",
        dense_model=ENSEMBLE_MODEL_NAMES,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    scores, contexts = retriever_rrf.retrieve(query, topk=3)


    # ============================
    # RRF (Validation Set)
    # ============================
    print("\n" + "=" * 60)
    print(f"RRF 테스트 (Validation Set, K={RRF_K} passage 기반 Hybrid)")
    print("=" * 60)

    # RRF는 alpha가 필요 없으므로 기본 초기화만 합니다.
    retriever_rrf = HybridRetrievalRRF(
        tokenize_fn=tokenizer.tokenize,
        data_path="../data",
        dense_model=ENSEMBLE_MODEL_NAMES,
    )

    # Validation 테스트
    val_df_rrf = retriever_rrf.retrieve(dataset["validation"], topk=5)

    # passage 기반이니까 "정답 텍스트가 context 안에 포함됐는지"로 hit 정의
    if "answers" in val_df_rrf.columns:
        def has_answer(row):
            answers = row["answers"]["text"]  # dict 형태: {"text": [...], "answer_start": [...]}

            if isinstance(answers, str):
                answers_list = [answers]
            else:
                answers_list = list(answers)

            ctx = row["context"]
            # 정답 후보 중 하나라도 context 안에 포함되면 hit
            return any(a and a in ctx for a in answers_list)

        val_df_rrf["hit"] = val_df_rrf.apply(has_answer, axis=1)
        accuracy_rrf = val_df_rrf["hit"].mean()

        print(f"✅ Hybrid RRF (K={RRF_K}): {accuracy_rrf:.4f}")
        
    
    # ============================
    # 기존 Alpha 튜닝 코드 (비교용)
    # ============================
    alphas = [0.1, 0.2, 0.3, 0.4]

    print("\n" + "=" * 60)
    print("Alpha 튜닝 (Validation Set, passage 기반 Hybrid)")
    print("=" * 60)

    for alpha in alphas:
        print(f"\n[α={alpha}] Testing...")

        retriever_alpha = HybridRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path="../data",
            dense_model=ENSEMBLE_MODEL_NAMES, # 추가: ENSEMBLE_MODEL_NAMES 명시
            alpha=alpha,
        )

        # Validation 테스트
        val_df_alpha = retriever_alpha.retrieve(dataset["validation"], topk=5)

        # passage 기반이니까 "정답 텍스트가 context 안에 포함됐는지"로 hit 정의
        if "answers" in val_df_alpha.columns:
            def has_answer_alpha(row):
                answers = row["answers"]["text"]
                if isinstance(answers, str):
                    answers_list = [answers]
                else:
                    answers_list = list(answers)

                ctx = row["context"]
                return any(a and a in ctx for a in answers_list)

            val_df_alpha["hit"] = val_df_alpha.apply(has_answer_alpha, axis=1)
            accuracy_alpha = val_df_alpha["hit"].mean()

            print(f"✅ Hybrid (α={alpha}): {accuracy_alpha:.4f}")

