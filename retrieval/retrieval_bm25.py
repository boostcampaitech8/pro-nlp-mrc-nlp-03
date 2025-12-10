# retrieval_bm25.py
import json
import os
import pickle
from typing import List, Optional, Tuple, Union
from rank_bm25 import BM25Okapi
import numpy as np
from datasets import Dataset
from tqdm.auto import tqdm
import pandas as pd

class BM25Retrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: str = "./data",
        context_path: str = "wikipedia_passages_256_128.json",
    ):
        """
        BM25 기반 Retrieval

        Arguments:
            tokenize_fn: 토크나이저 함수 (예: tokenizer.tokenize)
            data_path: 데이터 경로
            context_path: Wikipedia 문서 파일명
        """
        self.tokenize_fn = tokenize_fn
        self.data_path = data_path

        # Wikipedia 문서 로드
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Loaded {len(self.contexts)} unique contexts")

        self.bm25 = None

    def get_sparse_embedding(self):
        """
        BM25 인덱스 생성
        """
        pickle_name = "bm25_index.bin"
        bm25_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(bm25_path):
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            print("BM25 index loaded from pickle")
        else:
            print("Building BM25 index...")
            # 모든 문서를 토크나이즈
            tokenized_corpus = [
                self.tokenize_fn(doc) for doc in tqdm(self.contexts, desc="Tokenizing")
            ]
            self.bm25 = BM25Okapi(tokenized_corpus)

            with open(bm25_path, "wb") as f:
                pickle.dump(self.bm25, f)
            print("BM25 index saved")

    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: int = 5
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Query에 대해 관련 문서 검색

        Arguments:
            query_or_dataset: 단일 질문(str) 또는 Dataset
            topk: 상위 k개 문서 반환

        Returns:
            단일 query: (scores, contexts)
            Dataset: DataFrame with retrieved contexts
        """
        assert self.bm25 is not None, "get_sparse_embedding()을 먼저 실행하세요"

        if isinstance(query_or_dataset, str):
            # 단일 query 처리
            doc_scores, doc_indices = self._get_relevant_doc(query_or_dataset, topk)

            print(f"\n[Query] {query_or_dataset}\n")
            for i in range(topk):
                print(f"Top-{i+1} (score: {doc_scores[i]:.4f})")
                print(f"{self.contexts[doc_indices[i]][:200]}...\n")

            return doc_scores, [self.contexts[idx] for idx in doc_indices]

        elif isinstance(query_or_dataset, Dataset):
            # Dataset 처리
            return self._retrieve_bulk(query_or_dataset, topk)

    def _get_relevant_doc(self, query: str, k: int) -> Tuple[List, List]:
        """
        단일 query에 대한 BM25 검색
        """
        tokenized_query = self.tokenize_fn(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        # 상위 k개 인덱스
        sorted_indices = np.argsort(doc_scores)[::-1]
        doc_indices = sorted_indices[:k].tolist()
        doc_scores = doc_scores[sorted_indices[:k]].tolist()

        return doc_scores, doc_indices

    # _retrieve_bulk 함수만 수정
    def _retrieve_bulk(self, dataset: Dataset, topk: int) -> pd.DataFrame:
        total = []

        for example in tqdm(dataset, desc="BM25 Retrieval"):
            question = example["question"]
            doc_scores, doc_indices = self._get_relevant_doc(question, topk)

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

    # 설정
    dataset_name = "../data/train_dataset"
    model_name = "klue/bert-base"
    data_path = "../data"
    context_path = "wikipedia_documents.json"

    dataset = load_from_disk(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    retriever = BM25Retrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=data_path,
        context_path=context_path,
    )

    retriever.get_sparse_embedding()

    # 테스트 1: 단일 query
    print("\n" + "="*50)
    print("단일 Query 테스트")
    print("="*50)
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    scores, contexts = retriever.retrieve(query, topk=3)

    # 테스트 2: validation dataset
    print("\n" + "="*50)
    print("Validation Set 검색")
    print("="*50)
    val_df = retriever.retrieve(dataset["validation"], topk=5)

    # 정확도 측정
    if "hit" in val_df.columns:
        top_k_accuracy = val_df["hit"].sum() / len(val_df)
        print(f"\n✅ Top-5 Retrieval Accuracy: {top_k_accuracy:.4f}")
        print(f"   정답 찾은 개수: {val_df['hit'].sum()} / {len(val_df)}")

        # MRR 계산 (Mean Reciprocal Rank)
        def get_rank(row):
            if not row["hit"]:
                return 0
            try:
                rank = row["retrieved_contexts"].index(row["original_context"]) + 1
                return 1.0 / rank
            except:
                return 0

        val_df["reciprocal_rank"] = val_df.apply(get_rank, axis=1)
        mrr = val_df["reciprocal_rank"].mean()
        print(f"   MRR (Mean Reciprocal Rank): {mrr:.4f}")