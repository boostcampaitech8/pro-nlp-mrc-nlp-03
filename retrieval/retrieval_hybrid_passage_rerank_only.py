#retrieval_hybrid_passage_rerank_only.py 

import numpy as np
from typing import List, Tuple, Union

from datasets import Dataset
import pandas as pd
from tqdm.auto import tqdm

from retrieval.retrieval_hybrid_passage import HybridRetrievalRRF as RRFHybridRetrieval 

import torch
from transformers import (
    AutoTokenizer as HfAutoTokenizer,
    AutoModelForSequenceClassification,
)


class HybridRetrievalRerankRRF(RRFHybridRetrieval):
    """
    RRF Hybrid Retrieval (BM25 + Dense RRF)의 점수로 상위 k_candidate를 뽑고,
    Cross-Encoder reranker로 다시 정렬한 뒤 최종 top-k를 반환하는 리트리버.
    """

    def __init__(
        self,
        *args,
        use_rerank: bool = True,
        rerank_model_name: str = "Dongjin-kr/ko-reranker", 
        rerank_candidate_k: int = 20,   # RRF Hybrid에서 먼저 뽑을 개수
        device: str = None,
        rerank_batch_size: int = 4,    
        **kwargs,
    ):
        # super() 호출 시 RRFHybridRetrieval의 __init__이 실행되므로, 
        # 이제 self._get_relevant_doc()은 RRF 로직을 따릅니다.
        super().__init__(*args, **kwargs)

        self.use_rerank = use_rerank
        self.rerank_candidate_k = rerank_candidate_k
        self.rerank_batch_size = rerank_batch_size # 💡 [저장]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.use_rerank:
            print(f"[HybridRetrievalRerankRRF] Loading reranker: {rerank_model_name}")
            
            # 토크나이저 로드
            self.rerank_tokenizer = HfAutoTokenizer.from_pretrained(rerank_model_name)
            
            if self.rerank_tokenizer.pad_token is None:
                self.rerank_tokenizer.pad_token = self.rerank_tokenizer.eos_token
                print(f"   [Rerank Tokenizer] Set pad_token to eos_token: {self.rerank_tokenizer.pad_token}")

            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
                rerank_model_name
            ).to(self.device)
            self.rerank_model.eval()

            # 💡 [추가] 모델 config에도 pad_token_id를 설정하여 안전성 확보
            if self.rerank_model.config.pad_token_id is None:
                self.rerank_model.config.pad_token_id = self.rerank_tokenizer.pad_token_id


    # -----------------------------
    # Cross-Encoder Rerank (최종 수정)
    # -----------------------------
    def _rerank_passages(self, question: str, doc_indices: List[int]) -> List[int]:
        """
        Cross-encoder로 question + passage pairs를 점수화하여 doc_indices를 재정렬.
        - 배치 단위로 추론하여 메모리 OOM 방지
        - logits의 마지막 차원(예: 2)이 있는 경우 positive class (index 1)를 사용
        - 최종 scores 길이가 doc_indices 길이와 다르면 안전하게 보정
        """
        passages = [self.contexts[idx] for idx in doc_indices]
        num_passages = len(passages)
        all_scores = []

        # 빈 경우 빠르게 반환
        if num_passages == 0:
            return doc_indices

        # 배치 단위로 추론
        for i in tqdm(
            range(0, num_passages, self.rerank_batch_size),
            desc=f"Reranking (Batch Size: {self.rerank_batch_size})",
            leave=False,
        ):
            batch_passages = passages[i : i + self.rerank_batch_size]
            batch_questions = [question] * len(batch_passages)

            inputs = self.rerank_tokenizer(
                batch_questions,
                batch_passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.rerank_model(**inputs)
                logits = outputs.logits  # shape: (B, num_labels) or (B, 1) or (B,)

                # --- 안전한 score 추출 ---
                if logits is None:
                    # 예외적 상황: logits가 None이면 최소한 0점 할당
                    batch_scores = torch.zeros(len(batch_passages), device="cpu")
                else:
                    # multi-class (예: (B,2)) -> positive class 사용 (index 1)
                    if logits.dim() == 2 and logits.size(-1) >= 2:
                        # logits[:,1] 사용 (관련성 클래스)
                        batch_scores = logits[:, 1].cpu()
                    else:
                        # regression / single-logit: squeeze 후 확실히 1D로 변환
                        batch_scores = logits.view(-1).cpu()

            all_scores.append(batch_scores)

        # concat하여 (N,) 형태로 만듦
        if len(all_scores) == 0:
            return doc_indices

        final_scores = torch.cat(all_scores, dim=0)

        # 보장: 1D로 만들기
        final_scores = final_scores.reshape(-1)

        # ---- 방어: final_scores 길이와 doc_indices 길이 맞추기 ----
        if final_scores.size(0) != num_passages:
            # 로그 찍기 (logger 사용 가능하면 logger.warning으로)
            print(f"[Warning] rerank scores length ({final_scores.size(0)}) != num_passages ({num_passages}). Applying safe trim/pad.")

            if final_scores.size(0) > num_passages:
                # 초과하면 앞부분(또는 처음 num_passages)로 자름
                final_scores = final_scores[:num_passages]
            else:
                # 부족하면 -inf로 패딩하여 우선순위에서 뒤로 밀기
                pad_len = num_passages - final_scores.size(0)
                pad_tensor = torch.full((pad_len,), float("-inf"))
                final_scores = torch.cat([final_scores, pad_tensor], dim=0)

        # argsort (내림차순)
        sorted_indices_tensor = torch.argsort(final_scores, descending=True)

        # 안전하게 python int 리스트로 변환
        sorted_idx = [int(x) for x in sorted_indices_tensor.cpu().tolist()]

        # 방어: 인덱스 범위 확인 (여전히 혹시 모를 이상값 제거)
        sorted_idx = [i for i in sorted_idx if 0 <= i < num_passages]

        # 재정렬된 doc_indices 반환
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

        # 💡 [수정] 출력 메시지에 RRF 반영
        for example in tqdm(dataset, desc="Hybrid Retrieval (RRF) + Rerank"):
            question = example["question"]

            # 1단계: RRFHybridRetrieval 로 상위 rerank_candidate_k 개 뽑기
            k_candidate = self.rerank_candidate_k if self.use_rerank else topk
            # 💡 RRFHybridRetrieval의 _get_relevant_doc()이 호출되어 RRF 점수 기반으로 후보군 추출
            _, doc_indices = self._get_relevant_doc(question, k=k_candidate)

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