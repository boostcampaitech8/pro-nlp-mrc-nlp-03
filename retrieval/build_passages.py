# build_passages.py
import json
import os
from tqdm.auto import tqdm
from transformers import AutoTokenizer

DATA_PATH = "../../data"
DOC_PATH = "wikipedia_documents.json"
OUT_PATH = "wikipedia_passages_256_128.json"

# bge-m3에 맞추고 싶으면 이거
TOKENIZER_NAME = "BAAI/bge-m3"
# e5 기준으로 하고 싶으면
# TOKENIZER_NAME = "intfloat/multilingual-e5-base"

MAX_LEN = 256   # 청크 토큰 길이
STRIDE  = 128   # 겹치는 길이

def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    with open(os.path.join(DATA_PATH, DOC_PATH), "r", encoding="utf-8") as f:
        wiki = json.load(f)

    passages = {}
    new_id = 0

    for doc_id, doc in tqdm(wiki.items(), desc="Building passages"):
        text = doc["text"]

        # 토큰 단위로 슬라이딩 윈도우
        tokens = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )

        input_ids = tokens["input_ids"]
        offsets = tokens["offset_mapping"]

        n = len(input_ids)
        start = 0
        chunk_idx = 0

        while start < n:
            end = min(start + MAX_LEN, n)

            # 이 청크에 해당하는 원문 문자 범위
            char_start = offsets[start][0]
            char_end   = offsets[end - 1][1]

            passage_text = text[char_start:char_end]

            passage_id = f"{doc_id}_{chunk_idx}"
            passages[passage_id] = {
                "doc_id": doc_id,
                "passage_id": chunk_idx,
                "text": passage_text
            }

            chunk_idx += 1
            start += (MAX_LEN - STRIDE)  # stride 만큼 겹치게 이동

    out_file = os.path.join(DATA_PATH, OUT_PATH)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(passages)} passages to {out_file}")


if __name__ == "__main__":
    main()
