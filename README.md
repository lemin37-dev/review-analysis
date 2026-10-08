# 아마존 리뷰 감성 + 영→한 번역 실험 (노트북 → 로컬 프로젝트)

Colab 노트북 `amazon_sentiment_finetune`, `amazon_ko_translate_sentiment_v2`를 모듈과 CLI로 옮긴 것이다.
출력 폴더 구조(`voc_amazon_finetune/`, `voc_amazon_ko/`)는 노트북과 같아서 기존 결과를 그대로 쓴다.

## 설치
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt        # 로컬(CPU): 평가·리포트·Gemini
.venv/Scripts/python -m pip install -r requirements-gpu.txt    # GPU 머신/Colab: 번역 모델·파인튜닝
```
비밀값은 `.env`(`.env.example` 참고)에 둔다: `GEMINI_KEY`, `PAPAGO_CLIENT_ID/SECRET`, `HF_TOKEN`.

## 실행
```bash
python run.py check                       # 분할 재현, 번역 캐시 상태 (빈 번역 많은 후보 표시)
python run.py translate --model Rosetta-4B --limit 5   # 소량 검증 (translations_smoke/에 저장, GPU)
python run.py compare                     # 캐시된 번역으로 비교표 (--translate면 없는 후보도 번역)
python run.py train --only KcELECTRA --limit 200       # 소량 학습 검증 (GPU)
python run.py report                      # 요약표, 예측표, dashboard.html (GPU 불필요)
```
설정은 `review_analysis/config.py` 하나에 모았다(노트북 '1. 실험 설정' 셀과 같은 값).

## 노트북 v2 문제의 처리
| 문제 | 원인(노트북 로그 기준) | 처리 |
|---|---|---|
| Rosetta 빈 출력 | 300건 전부 빈 문자열. 원인은 미확정이나 Gemma 계열 fp16 불안정이 유력 | 후보 설정 `dtype='bf16'`, 빈 출력이면 원본 생성 문자열을 로그로 남김, 빈 번역은 캐시로 인정하지 않고 재번역 |
| EXAONE 로드 실패 | `create_causal_mask(input_embeds)` TypeError, 원격 코드와 새 transformers 불일치 | `requirements-gpu.txt`에서 transformers 상한 `<4.54` (GPU에서 미검증) |
| iris/llama3 로드 실패 | `No space left on device` (Colab 디스크) | 모델 하나가 끝날 때마다 HF 캐시 삭제, 받기 전 남은 디스크 경고 |
| 역번역 chrF 없음 | M2M100-1.2B 다운로드 중 디스크 부족 | 위 디스크 대응 + 빈 번역은 `.`로 대체해 역번역 |
| 중국어 혼입 미검출 | 한글 비율 50%만 검사 | `text.failure_reason`에 한자·가나 검사 추가(한글 비율 검사보다 먼저) |

## 현재 상태
- GPU가 없는 PC라서 번역 모델 실행과 파인튜닝은 아직 돌려 보지 못했다. 분할 재현, 요약표·비교표·대시보드 재생성은 기존 결과와 일치함을 확인했다.
- `amazon_llm_test_gemini.ipynb`는 아직 옮기지 않았다.

## 추후 고려사항
- **Gemini API 번역을 일 1회/시간 단위 배치로 운영할 때**: 현재 설정(20건 묶음, 요청 간 4초 대기)에서 2만 건은 대기만 약 67분. 주기당 물량, 요금제(무료 등급 요청 제한), 모델 이름 고정과 변경 점검, 자사 리뷰 사용 시 외부 전송 검토가 필요하다. 번역만 Gemini로 하고 감성 분류는 로컬 모델로 두면 API 호출이 줄어든다. 결정은 보류했고, 필요해지면 새 리뷰만 번역하는 증분 배치 명령을 `run.py`에 추가한다.
- **번역 모델**: v2는 NHNDQ-NLLB-en2ko를 사용했다(NLLB 계열은 비상업 라이선스). 서비스용이면 M2M100-1.2B, Gugugo-7B, Gemini 중에서 다시 정한다.

## 속성 기반 감성분석(ABSA) 시험 (의류 한정, 3단계, Gemini)
```bash
python run.py absa lang     # 영어 원문 vs 한글 번역문의 속성·감성 일관성 (300건)
python run.py absa rerun    # 같은 영어 입력을 두 번 넣어 Gemini 자체 흔들림 측정
python run.py absa synth    # 한글 합성 리뷰 200건 생성 + 정답 대비 정확도 (낙관적 수치)
```
- 라벨 기준과 평가셋 작성법: `docs/absa-label-guide.md`, 템플릿 `docs/absa_eval_template.csv`
- 결과 파일: `voc_amazon_ko/absa_pilot/` (Gemini 호출 결과 캐시)
- 한계: 번역문·합성 리뷰 기준이라 실제 한국인 리뷰에서의 성능은 아직 모른다. 직접 작성·검수한 평가셋이 필요하다.

### ABSA 후속 (1차 확정 반영)
```bash
python run.py absa synth --n 300        # 합성 한글 리뷰 생성 + 재추출 검증, 불일치 항목은 voc_amazon_ko/absa_pilot/synth_review_queue.v1.csv
python run.py absa eval --csv 경로.csv   # 사람이 작성한 평가셋(docs/absa_eval_template.csv 형식)을 Gemini 추출과 비교
python run.py absa-train --en 2000       # (GPU/Colab) KLUE-RoBERTa 다중 헤드 증류. 이 PC에서는 미검증
```
- 프롬프트가 바뀌면 `review_analysis/absa.py`의 `VERSION`을 올린다(캐시 파일 이름에 버전이 붙는다). v0 파일은 가이드 확정 전 시험 결과다.

## 전체 결과 화면
```bash
python run.py overview     # voc_amazon_ko/overview.html 생성 (저장된 결과만 읽음, GPU·API 불필요)
```
브라우저로 `voc_amazon_ko/overview.html`을 연다. 번역 모델 비교, 감성 모델(별점 5단계 + 긍/부정 점수), ABSA 시험, 추후 진행사항이 한 페이지에 있다. 별점 모델 상세(혼동행렬, 예측표)는 같은 폴더의 `dashboard.html`.

## 추후 진행사항
`overview.html` 4장과 같은 목록이다(코드: `review_analysis/overview.py`의 `NEXT_STEPS`).
- 데이터·평가: 한글 평가셋 200~300건 작성·검수, 합성 리뷰 불일치 확인, `absa eval`로 실제 말투 성능 확인, AI Hub 조건 확인과 문의
- 모델 학습(GPU/Colab): `absa-train` 소량 동작 확인 → 영어+한글 합본 학습, Rosetta/EXAONE/iris/llama3 재비교와 역번역 chrF, clapAI 영어 재실행
- 번역·운영 결정: 서비스용 번역 모델 선택, Gemini 배치 운영 여부, 2단계 번역·후보 선택 실험
