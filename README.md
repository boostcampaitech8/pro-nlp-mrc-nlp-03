# Korean MRC with Negative Passage Training

한국어 기계독해(MRC)를 위한 Negative Passage 학습 시스템

## 프로젝트 개요

본 프로젝트는 Open-Domain Question Answering을 위해 Negative Passage를 활용한 한국어 MRC 모델 학습 시스템입니다. 

### 주요 특징

- **Passage Indexing**: Wikipedia 문서를 Dense Encoder (BAAI/bge-m3) 토크나이저 기준 256 token / stride 128로 chunking하여 passage 단위로 인덱싱
- **Hybrid Retrieval**: BM25 + Dense Retrieval (BAAI/bge-m3) + Cross-Encoder Rerank
- **Negative Passage Training**: 정답이 포함되지 않은 passage를 함께 학습하여 모델 강건성 향상
- **Curriculum Learning**: Easy → Medium → Hard 단계별 학습
- **Position Bias 제거**: Passage 순서 랜덤화
- **Ensemble Voting**: Hard/Soft Voting을 통한 다중 모델 앙상블 지원

### 성능

| 모델 | Exact Match | F1 Score |
|------|-------------|----------|
| Baseline (v1) | 60.00% | 70.97% |
| Negative Passage (v2) | 64.58% | 76.09% |
| Curriculum Learning (v3) | **70.42%** | **78.17%** |

## 폴더 구조

```
korean-mrc-negative-passage/
├── data_preparation/          # 데이터셋 생성
│   ├── create_negative_passage_dataset.py
│   └── build_passages.py
├── retrieval/                 # 검색 모듈
│   ├── retrieval.py
│   ├── retrieval_bm25.py
│   ├── retrieval_dense.py
│   ├── retrieval_hybrid.py
│   ├── retrieval_hybrid_passage.py
│   └── retrieval_hybrid_passage_rerank_only.py
├── training/                  # 학습 모듈
│   ├── train.py
│   ├── trainer_qa.py
│   ├── arguments.py
│   └── utils_qa.py
├── inference/                 # 추론 모듈
│   ├── inference.py
│   ├── inference_bm25.py
│   └── inference_hybrid_passage_rerank_only.py
├── ensemble/                  # 앙상블 모듈
│   └── ensemble_voting.ipynb
├── scripts/                   # 실행 스크립트
│   └── create_negative_passage.sh
└── analysis/                  # 분석 도구
    ├── compare_predictions.py
    └── analyze_dataset.py
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 0. Wikipedia documents → passage corpus 생성
```bash
python data_preparation/build_passages.py
```

### 1. Negative Passage 데이터셋 생성

**기본 사용 (고정 개수)**
```bash
python data_preparation/create_negative_passage_dataset.py \
    --train_dataset_path ../data/train_dataset \
    --passages_path ../data/wikipedia_passages_256_128.json \
    --output_path ../data/train_dataset_negative_passage \
    --top_k_retrieval 100 \
    --rerank_top_k 5 \
    --alpha 0.7 \
    --use_rerank
```

**Curriculum Learning (Easy → Medium → Hard)**

Easy (3 passages):
```bash
python data_preparation/create_negative_passage_dataset.py \
    --curriculum_mode easy \
    --output_path ../data/train_dataset_easy
```

Medium (5 passages):
```bash
python data_preparation/create_negative_passage_dataset.py \
    --curriculum_mode medium \
    --output_path ../data/train_dataset_medium
```

Hard (7 passages):
```bash
python data_preparation/create_negative_passage_dataset.py \
    --curriculum_mode hard \
    --output_path ../data/train_dataset_hard
```

### 2. 모델 학습

**단일 스테이지 학습**
```bash
python -m training.train \
    --model_name_or_path HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned \
    --dataset_name ../data/train_dataset_negative_passage \
    --output_dir ../models/reader_negative_passage \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --fp16
```

**Curriculum Learning (순차 학습)**
```bash
# Stage 1: Easy
python -m training.train \
    --model_name_or_path HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned \
    --dataset_name ../data/train_dataset_easy \
    --output_dir ../models/curriculum_stage1_easy \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16

# Stage 2: Medium (이전 모델에서 시작)
python -m training.train \
    --model_name_or_path ../models/curriculum_stage1_easy \
    --dataset_name ../data/train_dataset_medium \
    --output_dir ../models/curriculum_stage2_medium \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16

# Stage 3: Hard
python -m training.train \
    --model_name_or_path ../models/curriculum_stage2_medium \
    --dataset_name ../data/train_dataset_hard \
    --output_dir ../models/curriculum_stage3_hard \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16
```

### 3. 추론

```bash
python -m inference.inference_hybrid_passage_rerank_only \
    --model_name_or_path ../models/curriculum_stage3_hard \
    --dataset_name ../data/test_dataset \
    --output_dir ../outputs/predictions
```

### 4. 앙상블 (Ensemble Voting)

**Hard Voting**: 다수결 투표 방식으로 가장 많이 예측된 답변을 선택

**Soft Voting**: 문자열 유사도를 기반으로 가중치를 부여하여 가장 높은 점수의 답변을 선택

**앙상블 통계**:
- 3개 모델 모두 일치: 70.17%
- 2개 모델 일치: 26.50%
- Hard voting과 Soft voting 차이: 약 3.33%의 케이스에서 다른 결과 도출

## 주요 개선 사항

### v3 (Curriculum Learning)
1. **Passage 순서 랜덤화**: Position bias 제거
2. **Hard Negative 품질 관리**: Score 기반 필터링
3. **Tokenizer 일관성**: Reader 모델과 동일한 tokenizer 사용
4. **Curriculum Learning**: 점진적 난이도 증가 (3 → 5 → 7 passages)

### v4 (Ensemble)
5. **Ensemble Voting 추가**: Hard Voting과 Soft Voting을 통한 다중 모델 앙상블
   - Hard Voting: 다수결 투표 방식
   - Soft Voting: 문자열 유사도 기반 가중치 투표
   - 3개 모델 앙상블 시 70%의 완전 일치율 확인

## 데이터셋

- **학습 데이터**: KorQuAD 1.0 (3,952 examples)
- **검증 데이터**: 240 examples
- **Wikipedia Passages**: bge-m3 tokenizer 기준 256 token / stride 128 chunk

## 모델

- **Reader**: HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned
- **Dense Retrieval**: BAAI/bge-m3
- **Reranker**: Dongjin-kr/ko-reranker

## 라이선스

MIT License

## 참고자료

- [KorQuAD 1.0](https://korquad.github.io/)
- [KLUE Benchmark](https://klue-benchmark.com/)




