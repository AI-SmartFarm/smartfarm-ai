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

## 중증도 판정 모델

- **모델**: 질병별 ResNet18 분류기 9개 (`models/severity-v2/model_{질병코드}.pt`, 각 약 45MB, Git LFS) — 탐지된 병반을 잘라 초기/중기/말기 판정
- **정확도**: 미확인 식물 기준 질병별 47~84%, 가중 평균 69.4% (`models/severity-v2/_summary.json`)
- 검증 정확도 0.6 미만인 질병(현재 고추점무늬병, 코드 4)은 응답에 `low_confidence: true`로 표시 — 앱에서는 참고용으로만 보여줄 것
- 모델 파일이 없는 질병은 중증도만 생략되고 진단은 그대로 진행

## 사용법

```bash
pip install rfdetr torch torchvision pillow supervision

python scripts/full_pipeline.py --image path/to/photo.jpg --crop tomato
```

와이드 앵글(고정 카메라) 샷처럼 병변이 프레임에서 작게 나오는 경우 `--tiles 3` 옵션 사용.

## 디렉터리 구조

- `models/rfdetr-s-synthetic-v5/` — 탐지 모델 체크포인트 + 클래스 매핑(`categories.json`)
- `models/severity-v2/` — 질병별 중증도 모델 9개 + 검증 결과(`_summary.json`)
- `scripts/` — 학습(`train_rfdetr.py`), 단독 추론(`infer_rfdetr.py`), 전체 파이프라인(`full_pipeline.py`)
- `data/disease_knowledge.json` — 질병별 원인/증상/예방·대응 지식베이스 (9개 클래스 전체 수록)

## AI 서버 실행과 외부 공개

```bash
pip install -r requirements.txt
API_KEY=<임의의 긴 문자열> uvicorn api:app --app-dir scripts --host 127.0.0.1 --port 8000
```

- `API_KEY`를 설정하면 `/diagnose`가 `X-API-Key` 헤더를 요구한다 (없거나 틀리면 401). `/health`는 인증 없이 열려 있다. `GET /ping`은 키를 검사하고 `{"status":"ok"}`를 돌려줘서, 호출하는 쪽이 키가 맞는지 확인하는 데 쓴다.
- 배포된 백엔드가 호출해야 하면 터널로 연다: `cloudflared tunnel --url http://127.0.0.1:8000` — 출력되는 `https://….trycloudflare.com` 주소가 백엔드의 `AI_SERVICE_URL`이 된다.
- 계정 없는 퀵 터널은 재시작할 때마다 주소가 바뀌고 가동을 보장하지 않는다. 터널을 열 때는 반드시 `API_KEY`를 설정한다.
