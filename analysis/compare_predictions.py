"""
두 예측 결과 CSV 파일 비교 분석
"""

# 파일 읽기
def read_predictions(file_path):
    predictions = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                predictions[parts[0]] = parts[1]
    return predictions

print("=" * 80)
print("예측 결과 비교 분석")
print("=" * 80)

# 두 파일 읽기
preds_v1 = read_predictions("outputs/predictions_negative_passage/predictions_submit.csv")
preds_v2 = read_predictions("outputs/predictions_negative_passage_v2/predictions_submit.csv")

print(f"\n1. 기본 정보")
print(f"   - v1 (EM 60.00%, F1 70.97%): {len(preds_v1)} 개")
print(f"   - v2 (EM 64.58%, F1 76.09%): {len(preds_v2)} 개")
print(f"   - 성능 향상: EM +{64.58-60.00:.2f}%, F1 +{76.09-70.97:.2f}%")

# 차이 분석
same_count = 0
diff_count = 0
differences = []

for qid in preds_v1:
    if qid in preds_v2:
        if preds_v1[qid] == preds_v2[qid]:
            same_count += 1
        else:
            diff_count += 1
            differences.append({
                'id': qid,
                'v1': preds_v1[qid],
                'v2': preds_v2[qid],
                'len_v1': len(preds_v1[qid]),
                'len_v2': len(preds_v2[qid])
            })

print(f"\n2. 예측 차이 통계")
print(f"   - 동일한 예측: {same_count} 개 ({same_count/len(preds_v1)*100:.2f}%)")
print(f"   - 다른 예측: {diff_count} 개 ({diff_count/len(preds_v1)*100:.2f}%)")

# 차이나는 예측들 출력
print(f"\n3. 다른 예측 샘플 (처음 30개):")
print("-" * 80)
for i, diff in enumerate(differences[:30]):
    print(f"\n[{i+1}] ID: {diff['id']}")
    print(f"  v1 ({diff['len_v1']}자): {diff['v1']}")
    print(f"  v2 ({diff['len_v2']}자): {diff['v2']}")

# 예측 길이 통계
all_len_v1 = [len(v) for v in preds_v1.values()]
all_len_v2 = [len(v) for v in preds_v2.values()]

avg_len_v1 = sum(all_len_v1) / len(all_len_v1)
avg_len_v2 = sum(all_len_v2) / len(all_len_v2)

print(f"\n4. 예측 답변 길이 통계")
print(f"   v1 평균 길이: {avg_len_v1:.2f} 자")
print(f"   v2 평균 길이: {avg_len_v2:.2f} 자")
print(f"   평균 길이 차이: {avg_len_v2 - avg_len_v1:+.2f} 자")

# 더 긴 답변 비율
longer_v1 = sum(1 for d in differences if d['len_v1'] > d['len_v2'])
longer_v2 = sum(1 for d in differences if d['len_v2'] > d['len_v1'])
same_len = sum(1 for d in differences if d['len_v1'] == d['len_v2'])

print(f"\n5. 차이나는 예측들의 길이 비교")
print(f"   v1이 더 긴 경우: {longer_v1} 개 ({longer_v1/diff_count*100:.2f}%)")
print(f"   v2가 더 긴 경우: {longer_v2} 개 ({longer_v2/diff_count*100:.2f}%)")
print(f"   같은 길이: {same_len} 개 ({same_len/diff_count*100:.2f}%)")

# 차이 저장
with open("outputs/prediction_differences.csv", 'w', encoding='utf-8') as f:
    f.write("id\tprediction_v1\tprediction_v2\tlen_v1\tlen_v2\n")
    for diff in differences:
        f.write(f"{diff['id']}\t{diff['v1']}\t{diff['v2']}\t{diff['len_v1']}\t{diff['len_v2']}\n")

print(f"\n6. 차이나는 예측들을 'outputs/prediction_differences.csv'에 저장했습니다.")

# 패턴 분석
print(f"\n7. 패턴 분석")
# v2가 더 구체적인지 확인
more_specific_v2 = 0
less_specific_v2 = 0

for diff in differences[:50]:  # 샘플로 50개만 분석
    v1, v2 = diff['v1'], diff['v2']
    # v2가 v1을 포함하거나 더 상세한 경우
    if v1 in v2 and len(v2) > len(v1):
        more_specific_v2 += 1
    elif v2 in v1 and len(v1) > len(v2):
        less_specific_v2 += 1

print(f"   (처음 50개 샘플 분석)")
print(f"   v2가 더 구체적/상세한 경우: {more_specific_v2} 개")
print(f"   v2가 덜 구체적/간결한 경우: {less_specific_v2} 개")

print("\n" + "=" * 80)
