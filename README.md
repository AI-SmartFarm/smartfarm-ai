# smartfarm-ai
GrowCare AI 모델 학습·추론 (Python)

## 병해충 진단 모델

- **모델**: RF-DETR-Small, 5개 작물(고추/딸기/상추/오이/토마토)의 14개 클래스(질병 9종 + 정상 5종) 탐지
- **체크포인트**: [`models/rfdetr-s-synthetic-v5/last_ema.pth`](models/rfdetr-s-synthetic-v5/last_ema.pth) (Git LFS로 관리, 123MB)
  - 마지막 epoch 체크포인트 사용 (val mAP가 가장 높은 `checkpoint_best_ema.pth`가 아님) — 합성 배경 증강 데이터를 더 많이 학습해 모든 robustness 테스트에서 일반화 성능이 더 좋았음
  - 추론 해상도는 체크포인트에 저장되지 않으므로 항상 **640**으로 명시
  - confidence threshold **0.15** 권장 (미확인 식물 기준 질병 recall 99.6%, 정상 오탐 2.2%)
- **데이터셋 계보**: AIHub "071.시설 작물 질병 진단" → plant-disjoint/생육단계 계층화 분할 → 배경 스왑/모자이크 합성 증강으로 파인튜닝
- **아직 검증되지 않은 부분**: 실제 스마트팜 카메라 영상으로는 테스트되지 않음 (AIHub 근접 촬영 사진 도메인에서만 측정된 수치)

## 사용법

```bash
pip install rfdetr torch torchvision pillow supervision

python scripts/full_pipeline.py --image path/to/photo.jpg --crop tomato
```

와이드 앵글(고정 카메라) 샷처럼 병변이 프레임에서 작게 나오는 경우 `--tiles 3` 옵션 사용.

## 디렉터리 구조

- `models/rfdetr-s-synthetic-v5/` — 탐지 모델 체크포인트 + 클래스 매핑(`categories.json`)
- `scripts/` — 학습(`train_rfdetr.py`), 단독 추론(`infer_rfdetr.py`), 전체 파이프라인(`full_pipeline.py`)
- `data/disease_knowledge.json` — 질병별 원인/증상/예방·대응 지식베이스 (9개 클래스 전체 수록)
