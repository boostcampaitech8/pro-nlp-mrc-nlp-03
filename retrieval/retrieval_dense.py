# # retrieval_dense.py
# import json
# import os
# import pickle
# from typing import List, Optional, Tuple, Union

# import numpy as np
# import pandas as pd
# from datasets import Dataset
# from tqdm.auto import tqdm
# from sentence_transformers import SentenceTransformer


# class DenseRetrieval:
#     def __init__(
#         self,
#         data_path: str = "./data",
#         context_path: str = "wikipedia_passages_256_128.json",
#         model_name: str = "jhgan/ko-sroberta-multitask",
#         embedding_name: Optional[str] = None,
#         normalize: bool = True,
#     ):
#         """
#         SentenceTransformer 기반 Dense Retrieval

#         Arguments:
#             data_path: 데이터 경로
#             context_path: Wikipedia 문서 파일명
#             model_name: SentenceTransformer 모델 이름
#             embedding_name: 임베딩 저장 파일명 (None이면 model_name 기반으로 자동 생성)
#             normalize: 임베딩 L2 정규화 여부 (cosine 유사도용)
#         """
#         self.data_path = data_path
#         self.context_path = context_path
#         self.model_name = model_name
#         self.normalize = normalize

#         # Wikipedia 문서 로드 (BM25Retrieval과 동일한 방식)
#         with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
#             wiki = json.load(f)

#         # 중복 제거된 context 목록
#         self.contexts: List[str] = list(dict.fromkeys([v["text"] for v in wiki.values()]))
#         print(f"[DenseRetrieval] Loaded {len(self.contexts)} unique contexts")

#         # SentenceTransformer 로드
#         print(f"[DenseRetrieval] Loading dense model: {model_name}")
#         self.model = SentenceTransformer(model_name)

#         self.model.max_seq_length = 256  # BGE 설정


#         # 임베딩 파일 이름 설정
#         if embedding_name is None:
#             safe_model_name = model_name.replace("/", "_")
#             embedding_name = f"dense_emb_{safe_model_name}.npy"

#         self.embedding_path = os.path.join(self.data_path, embedding_name)
#         self.passage_embs: Optional[np.ndarray] = None

#     # def build_dense_embedding(self, batch_size: int = 64):
#     def build_dense_embedding(self, batch_size: int = 8):  # BGE 설정

#         """
#         위키 전체 문서에 대한 dense 임베딩 생성 및 로드
#         """
#         if os.path.isfile(self.embedding_path):
#             print(f"[DenseRetrieval] Loading passage embeddings from {self.embedding_path}")
#             self.passage_embs = np.load(self.embedding_path)
#             return

#         print("[DenseRetrieval] Building passage embeddings...")
#         # # SentenceTransformer는 내부적으로 batching + tqdm 옵션 지원
#         # embs = self.model.encode(
#         #     self.contexts,
#         #     batch_size=batch_size,
#         #     convert_to_numpy=True,
#         #     show_progress_bar=True,
#         # )

#         # passage prefix 적용 버전 E5, BGE
#         passages = [f"passage: {c}" for c in self.contexts]

#         embs = self.model.encode(
#             passages,
#             batch_size=batch_size,
#             convert_to_numpy=True,
#             show_progress_bar=True,
#         )

#         # cosine 유사도용 L2 정규화
#         if self.normalize:
#             norms = np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-12
#             embs = embs / norms

#         self.passage_embs = embs
#         np.save(self.embedding_path, self.passage_embs)
#         print(f"[DenseRetrieval] Saved passage embeddings to {self.embedding_path}")

#     def _encode_query(self, query: str) -> np.ndarray:
#         q = self.model.encode([query], convert_to_numpy=True)
#         if self.normalize:
#             # q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)

#             # E5, BGE
#             q = self.model.encode([f"query: {query}"], convert_to_numpy=True)

#         return q  # (1, dim)

#     def _get_relevant_doc(self, query: str, k: int) -> Tuple[List[float], List[int]]:
#         """
#         단일 query에 대한 dense 검색
#         """
#         assert self.passage_embs is not None, "build_dense_embedding()을 먼저 실행하세요"

#         q_emb = self._encode_query(query)  # (1, dim)
#         # passage_embs: (N, dim)
#         scores = np.dot(self.passage_embs, q_emb.T).squeeze(-1)  # (N,)

#         sorted_indices = np.argsort(scores)[::-1]
#         doc_indices = sorted_indices[:k].tolist()
#         doc_scores = scores[doc_indices].tolist()

#         return doc_scores, doc_indices

#     def retrieve(
#         self,
#         query_or_dataset: Union[str, Dataset],
#         topk: int = 5,
#     ) -> Union[Tuple[List[float], List[str]], pd.DataFrame]:
#         """
#         Query에 대해 관련 문서 검색

#         Arguments:
#             query_or_dataset: 단일 질문(str) 또는 Dataset
#             topk: 상위 k개 문서 반환

#         Returns:
#             단일 query: (scores, contexts)
#             Dataset: DataFrame with retrieved contexts
#         """
#         assert self.passage_embs is not None, "build_dense_embedding()을 먼저 실행하세요"

#         if isinstance(query_or_dataset, str):
#             doc_scores, doc_indices = self._get_relevant_doc(query_or_dataset, topk)

#             print(f"\n[Query] {query_or_dataset}\n")
#             for i in range(topk):
#                 print(f"Top-{i+1} (score: {doc_scores[i]:.4f})")
#                 print(f"{self.contexts[doc_indices[i]][:200]}...\n")

#             return doc_scores, [self.contexts[idx] for idx in doc_indices]

#         elif isinstance(query_or_dataset, Dataset):
#             return self._retrieve_bulk(query_or_dataset, topk)

#         else:
#             raise TypeError("query_or_dataset는 str 또는 Dataset 이어야 합니다.")

#     def _retrieve_bulk(self, dataset: Dataset, topk: int) -> pd.DataFrame:
#         """
#         Dataset 기반 일괄 검색
#         """
#         total = []

#         for example in tqdm(dataset, desc="Dense Retrieval"):
#             question = example["question"]
#             doc_scores, doc_indices = self._get_relevant_doc(question, topk)

#             retrieved_contexts = [self.contexts[idx] for idx in doc_indices]

#             tmp = {
#                 "question": question,
#                 "id": example["id"],
#                 "context": " ".join(retrieved_contexts),
#             }

#             if "context" in example.keys() and "answers" in example.keys():
#                 tmp["original_context"] = example["context"]
#                 tmp["answers"] = example["answers"]

#             total.append(tmp)

#         return pd.DataFrame(total)


# if __name__ == "__main__":
#     """
#     간단 테스트:
#       - 단일 query에 대해 top-3 출력
#       - validation set에 대해 Top-5 Retrieval Accuracy 측정
#     """
#     from datasets import load_from_disk

#     dataset_name = "../data/train_dataset"
#     data_path = "../data"
#     context_path = "wikipedia_documents.json"
#     model_name = "jhgan/ko-sroberta-multitask"  # 필요하면 e5, bge로 교체 가능

#     dataset = load_from_disk(dataset_name)

#     retriever = DenseRetrieval(
#         data_path=data_path,
#         context_path=context_path,
#         model_name=model_name,
#     )
#     retriever.build_dense_embedding(batch_size=64)

#     # 테스트 1: 단일 query
#     print("\n" + "=" * 50)
#     print("단일 Query 테스트 (Dense)")
#     print("=" * 50)
#     query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
#     scores, contexts = retriever.retrieve(query, topk=3)

#     # 테스트 2: validation dataset
#     print("\n" + "=" * 50)
#     print("Validation Set Dense Retrieval")
#     print("=" * 50)
#     val_df = retriever.retrieve(dataset["validation"], topk=5)

#     if "original_context" in val_df.columns:
#         val_df["hit"] = val_df.apply(
#             lambda row: row["original_context"] in row["context"], axis=1
#         )
#         top_k_accuracy = val_df["hit"].sum() / len(val_df)
#         print(f"\n✅ Top-5 Dense Retrieval Accuracy: {top_k_accuracy:.4f}")
#         print(f"   정답 찾은 개수: {val_df['hit'].sum()} / {len(val_df)}")

import json
import os
import pickle
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


# 💡 [추가] 모델별 Prefix를 관리하는 헬퍼 함수
def get_prefix(model_name: str, is_query: bool) -> str:
    """
    BGE, E5 계열 모델에만 'query:' 또는 'passage:' prefix를 적용합니다.
    ko-sroberta-multitask와 같은 모델은 빈 문자열을 반환합니다.
    """
    # 모델 이름이 BGE 또는 E5를 포함하는 경우 (대소문자 무시)
    if "bge" in model_name.lower() or "e5" in model_name.lower():
        return "query: " if is_query else "passage: "
    return ""


class DenseRetrievalEnsemble:
    """
    SentenceTransformer 기반 Dense Retrieval (Embedding Concatenation Ensemble)
    """
    def __init__(
        self,
        data_path: str = "/data/ephemeral/home/yunhui/data",
        context_path: str = "wikipedia_passages_256_128.json",
        model_names: List[str] = ["jhgan/ko-sroberta-multitask", "BAAI/bge-m3"],
        embedding_name: Optional[str] = None,
        normalize: bool = True,
    ):
        
        self.data_path = data_path
        self.context_path = context_path
        self.model_names = model_names
        self.normalize = normalize
        self.models: List[SentenceTransformer] = []

        # Wikipedia 문서 로드
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        # 중복 제거된 context 목록
        self.contexts: List[str] = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"[DenseRetrievalEnsemble] Loaded {len(self.contexts)} unique contexts")

        # 앙상블 모델 로드 및 차원 계산
        self.total_dim = 0
        print(f"[DenseRetrievalEnsemble] Loading {len(model_names)} dense models:")
        for name in self.model_names:
            print(f"  - Loading {name}")
            model = SentenceTransformer(name)
            model.max_seq_length = 256
            self.models.append(model)
            self.total_dim += model.get_sentence_embedding_dimension()
        
        print(f"[DenseRetrievalEnsemble] Total concatenated dimension: {self.total_dim}")


        # 임베딩 파일 이름 설정 (모든 모델 이름을 반영)
        if embedding_name is None:
            safe_names = "_".join([name.replace("/", "_") for name in self.model_names])
            embedding_name = f"dense_emb_ensemble_{safe_names}.npy"

        self.embedding_path = os.path.join(self.data_path, embedding_name)
        self.passage_embs: Optional[np.ndarray] = None


    def build_dense_embedding(self, batch_size: int = 512):
        """
        위키 전체 문서에 대한 Dense 임베딩을 모든 모델로 생성 후 연결
        """
        if os.path.isfile(self.embedding_path):
            print(f"[DenseRetrievalEnsemble] Loading passage embeddings from {self.embedding_path}")
            self.passage_embs = np.load(self.embedding_path)
            
            # 💡 [수정] 차원 불일치 시 경고 대신 오류 발생 (안정성 강화)
            if self.passage_embs.shape[1] != self.total_dim:
                raise ValueError(
                    f"Loaded embedding dimension ({self.passage_embs.shape[1]}) does not match "
                    f"expected concatenated dimension ({self.total_dim}). "
                    f"Please delete the file {self.embedding_path} and rebuild."
                )
            return

        print("[DenseRetrievalEnsemble] Building concatenated passage embeddings...")
        
        all_embs = []
        
        # 💡 [수정] 모델별로 다른 prefix를 적용하여 passages 인코딩
        for i, model in enumerate(self.models):
            model_name = self.model_names[i]
            print(f"  - Encoding passages with {model_name}")
            
            prefix = get_prefix(model_name, is_query=False)
            passages_with_prefix = [f"{prefix}{c}" for c in self.contexts]

            embs = model.encode(
                passages_with_prefix,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            all_embs.append(embs)
        
        # 임베딩 연결 (Concatenation)
        self.passage_embs = np.concatenate(all_embs, axis=1)

        # cosine 유사도용 L2 정규화 (연결된 최종 벡터에 대해 수행)
        if self.normalize:
            norms = np.linalg.norm(self.passage_embs, axis=-1, keepdims=True) + 1e-12
            self.passage_embs = self.passage_embs / norms

        np.save(self.embedding_path, self.passage_embs)
        print(f"[DenseRetrievalEnsemble] Saved passage embeddings to {self.embedding_path}")

    def _encode_query(self, query: str) -> np.ndarray:
        """
        단일 query를 모든 모델로 인코딩 후 연결
        """
        all_q_embs = []
        
        # 💡 [수정] 모델별로 다른 prefix를 적용하여 쿼리 인코딩
        for i, model in enumerate(self.models):
            model_name = self.model_names[i]
            prefix = get_prefix(model_name, is_query=True)
            
            queries = [f"{prefix}{query}"] 

            q_emb = model.encode(queries, convert_to_numpy=True)
            all_q_embs.append(q_emb)

        # 쿼리 벡터 연결
        q_final = np.concatenate(all_q_embs, axis=1) # (1, Final_Dim)

        if self.normalize:
            # L2 정규화
            norms = np.linalg.norm(q_final, axis=-1, keepdims=True) + 1e-12
            q_final = q_final / norms
            
        return q_final # (1, Final_Dim)


    def _get_relevant_doc(self, query: str, k: int) -> Tuple[List[float], List[int]]:
        """
        단일 query에 대한 dense 검색 (연결된 임베딩 사용)
        """
        assert self.passage_embs is not None, "build_dense_embedding()을 먼저 실행하세요"

        q_emb = self._encode_query(query)  # (1, Final_Dim)
        # 내적 (Dot Product)을 통한 유사도 계산
        scores = np.dot(self.passage_embs, q_emb.T).squeeze(-1)  # (N,)

        sorted_indices = np.argsort(scores)[::-1]
        doc_indices = sorted_indices[:k].tolist()
        doc_scores = scores[doc_indices].tolist()

        return doc_scores, doc_indices

    # retrieve, _retrieve_bulk 메서드는 변경 없음

    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: int = 5,
    ) -> Union[Tuple[List[float], List[str]], pd.DataFrame]:
        """
        Query에 대해 관련 문서 검색
        """
        assert self.passage_embs is not None, "build_dense_embedding()을 먼저 실행하세요"

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self._get_relevant_doc(query_or_dataset, topk)

            print(f"\n[Query] {query_or_dataset}\n")
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
        Dataset 기반 일괄 검색
        """
        total = []

        for example in tqdm(dataset, desc="Dense Ensemble Retrieval"):
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
    """
    간단 테스트:
      - 단일 query에 대해 top-3 출력
      - validation set에 대해 Top-5 Retrieval Accuracy 측정
    """
    from datasets import load_from_disk

    dataset_name = "/data/ephemeral/home/yunhui/data/train_dataset"
    data_path = "/data/ephemeral/home/yunhui/data"
    context_path = "wikipedia_passages_256_128.json" 
    
    ENSEMBLE_MODEL_NAMES = ["jhgan/ko-sroberta-multitask", "BAAI/bge-m3"]

    dataset = load_from_disk(dataset_name)

    retriever = DenseRetrievalEnsemble(
        data_path=data_path,
        context_path=context_path,
        model_names=ENSEMBLE_MODEL_NAMES,
    )
    # build_dense_embedding 실행 전에, 파인튜닝된 모델을 사용한다면
    # 기존에 저장된 임베딩 파일(dense_emb_ensemble_...npy)을 삭제해야 합니다!
    retriever.build_dense_embedding(batch_size=256) 

    # 테스트 1: 단일 query
    print("\n" + "=" * 50)
    print("단일 Query 테스트 (Dense Ensemble)")
    print("=" * 50)
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    scores, contexts = retriever.retrieve(query, topk=3)

    # 테스트 2: validation dataset
    print("\n" + "=" * 50)
    print("Validation Set Dense Ensemble Retrieval")
    print("=" * 50)
    val_df = retriever.retrieve(dataset["validation"], topk=5)

    if "original_context" in val_df.columns:
        val_df["hit"] = val_df.apply(
            lambda row: row["original_context"] in row["context"], axis=1
        )
        top_k_accuracy = val_df["hit"].sum() / len(val_df)
        print(f"\n✅ Top-5 Dense Ensemble Retrieval Accuracy: {top_k_accuracy:.4f}")
        print(f"   정답 찾은 개수: {val_df['hit'].sum()} / {len(val_df)}")