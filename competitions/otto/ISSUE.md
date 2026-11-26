# Otto — 막혔던 것들

작업 중 걸린 환경·도구 문제 기록. 2026-11-20 ~ 11-26.

| # | 이슈 | 해결 |
|---|---|---|
| 1 | [AutoGluon segfault (macOS libomp 중복 로드)](#1-autogluon-segfault--macos-libomp-중복-로드) | `OMP_NUM_THREADS=1` |
| 2 | [`autogluon.tabular` 만 설치하면 부스팅 백엔드가 없다](#2-autoguontabular-만-설치하면-부스팅-백엔드가-없다) | `[all]` extras |
| 3 | [AutoGluon 1.6.3 × pandas 3.x 비호환](#3-autogluon-163--pandas-3x-비호환) | `pandas<3` |
| 4 | [(사내) Fasoo DRM 이 패키지 메타데이터를 암호화](#4-사내-fasoo-drm-이-패키지-메타데이터를-암호화) | 정상 파일 직접 덮어쓰기 |

**공통점 — 넷 다 exit code 가 0이다.** 로그를 열기 전에는 성공처럼 보인다.

---

# 1. AutoGluon segfault — macOS libomp 중복 로드

## 요약

AutoGluon `TabularPredictor.fit()` 은 **매번 정상 완료**되는데,
그 뒤 **저장된 모델을 다시 로드하는 모든 동작에서 Segmentation fault** 로 프로세스가 죽는다.
`predict_proba()`, `refit_full()`, `predictor.info()` 가 전부 해당한다.
학습 결과(리더보드)는 얻을 수 있으나 **예측·제출 파일을 만들 수 없다.**

## 환경

| | |
|---|---|
| OS | macOS (Darwin 25.6.0), Apple Silicon |
| Python | 3.11.13 (homebrew) |
| autogluon.tabular | 1.6.3 (`[all]` extras) |
| pandas | 2.3.3 (3.x 는 별도 비호환, 아래 참고) |
| torch / lightgbm / xgboost / catboost | 2.x / 4.7.0 / 3.2.0 / 1.2.10 |
| 데이터 | 61,878 행 × 93 피처, 9클래스 / test 144,368 행 |
| 메모리 | 24GB (사용량 2.3GB, 여유 충분) |
| GPU | 없음 (`gpus=0`, CPU 학습) |

## 증상

```
AutoGluon training complete, total runtime = 603.89s ... Best model: WeightedEnsemble_L3
FIT_DONE
FITTED=21                      # 리더보드 정상 출력
  ↓
Fatal Python error: Segmentation fault      # predict_proba() 호출
```

셸 exit code 는 **0** 으로 나온다. marimo 로 실행하면
"Export was successful, but some cells failed to execute" 만 뜨고 성공처럼 보인다.
**로그를 열기 전에는 실패를 알 수 없다.**

## 시도한 것과 결과

| # | 변경 | 결과 |
|---|---|---|
| 1 | 원본 설정 (`best_quality`, DyStack on) | DyStack 예비 fit 성공 → **본 학습 재진입 시 segfault** |
| 2 | 1 재시도 | **동일 지점에서 재현** (비결정적 아님) |
| 3 | `dynamic_stacking=False` + `num_stack_levels=1` | 학습·리더보드 성공 → **`predictor.info()` 에서 segfault** |
| 4 | 3 + `excluded_model_types=['FASTAI','NN_TORCH']` + `num_bag_folds=5` | 학습 성공(트리 21개) → **`refit_full()` 에서 segfault** |
| 5 | 4 + `fold_fitting_strategy='sequential_local'` (ray off) | **학습 중 첫 fold 에서 사망** (더 악화) |
| 6 | 4 + 샌드박스 해제 (`sysctl` 접근 허용) | 학습·리더보드 성공 → **`predict_proba()` 에서 segfault** |

### 배제된 가설

- **torch/신경망**: `FASTAI`·`NN_TORCH` 를 완전히 제외해도 재현됨.
  크래시 직전 `torch.jit.script` FutureWarning 이 찍혀서 의심했으나 인과 아님
- ~~**OpenMP 중복 로드**~~ → **이것이 원인이었다.** 아래 「원인」 절 참고.
  최소 재현(LightGBM 학습 → `torch.jit.script`)이 통과해서 한 번 배제했으나,
  **import 순서가 우연히 안전한 쪽이었다.** 알려진 이슈다
- **메모리 부족**: 24GB 중 2.3GB 사용. OOM 이면 `Killed`/`MemoryError` 가 나야 함
- **ray**: 끄면 오히려 더 일찍 죽음
- **샌드박스/`sysctl` 차단**: 해제해도 동일

### 공통 패턴

```
fit()  ───────────────► 항상 성공 (603초, 모델 21개, 리더보드 정상)
  │
  └─► 저장된 모델 재로드가 필요한 동작 ──► 100% segfault
        predict_proba() / refit_full() / info() / 본 학습 재진입
```

별도 프로세스에서 `TabularPredictor.load()` 후 예측을 시도한 경우,
segfault 대신 **CPU 0%·RSS 1.2MB 상태로 2시간 44분 행(hang)** 되었다.

## 원인 — macOS libomp 중복 로드 (확인됨)

PyTorch·LightGBM 양쪽에 같은 증상의 이슈가 열려 있다.

- [pytorch#191933 — Segmentation fault on MacOS when running alongside LightGBM due to libomp incompatibility](https://github.com/pytorch/pytorch/issues/191933)
- [LightGBM#6595 — SegFault on MacOS when pytorch is installed](https://github.com/lightgbm-org/LightGBM/issues/6595)
- [pytorch#191971 — macOS: Detect conflicting OpenMP runtimes](https://github.com/pytorch/pytorch/pull/191971)

> 한 프로세스에서 torch 와 lightgbm 을 import 하면 **libomp.dylib 두 개가 로드되고
> libomp 내부에서 SIGSEGV 로 죽는다.** 두 런타임이 서로의 워커 스레드 상태를 오염시킨다.

macOS 휠마다 OpenMP 출처가 다르다.

```
torch, scikit-learn  → 자체 번들 libomp 포함
lightgbm, xgboost    → homebrew libomp 를 찾음
```

PyTorch 2.13 부터 `torch/lib/libomp.dylib` 의 install name 이
`@rpath/libomp.dylib` → `/opt/llvm-openmp/lib/libomp.dylib` 로 바뀌어
dyld 가 같은 라이브러리로 인식하지 못하고 **두 번 로드**한다.

실제 프로세스에서 확인된 것 (3개):

```
sklearn/.dylibs/libomp.dylib
torch/lib/libomp.dylib
/opt/homebrew/Cellar/libomp/20.1.7/lib/libomp.dylib
```

**모델 재로드에서만 터진 이유**도 이것으로 설명된다. 학습은 단일 경로로 끝나지만,
저장된 모델을 다시 로드할 때 워커 스레드를 새로 띄우면서 오염된 상태를 건드린다.

### `KMP_DUPLICATE_LIB_OK=TRUE` 는 쓰면 안 된다

- [pytorch#193520 — KMP_DUPLICATE_LIB_OK=TRUE with a second libomp causes intermittent autograd deadlock and MPS sync hangs](https://github.com/pytorch/pytorch/issues/193520)

**데드락과 행(hang)을 유발**한다. 별도 프로세스에서 `load()`+predict 가
CPU 0% 로 2시간 44분 멈춘 것이 이 증상으로 보인다.

### 알려진 해결책

| 방법 | 효과 | 비용 |
|---|---|---|
| `OMP_NUM_THREADS=1` | 크래시 방지 확인됨 | 단일 스레드 → 느려짐 |
| import 순서를 torch 먼저로 | 동작 보고됨 | AutoGluon 내부 순서라 제어 어려움 |
| libomp 하나로 통일 (`install_name_tool` 재링크) | 근본 해결 | 수작업, 패키지 업데이트 시 재발 |
| conda-forge 빌드 사용 | OpenMP 를 내부에서 해결 | 환경 재구축 |
| Linux / Docker | 확실 | 환경 전환 |


---

## 얻은 결과 (참고용 — 크래시 전까지는 리더보드가 나온다)

크래시 전까지 리더보드는 정상적으로 나온다. 600초, 8-fold bagging, 2층 스태킹:

| 모델 | val log_loss |
|---|---|
| WeightedEnsemble_L3 | **0.41685** |
| LightGBMXT_BAG_L2 | 0.42241 |
| XGBoost_BAG_L2 | 0.42581 |
| LightGBM_BAG_L2 | 0.42782 |
| CatBoost_BAG_L2 | 0.43609 |
| NeuralNetFastAI_BAG_L2 | 0.43927 |
| LightGBM_BAG_L1 | 0.46851 |
| XGBoost_BAG_L1 | 0.47466 |
| LightGBMXT_BAG_L1 | 0.48058 |
| CatBoost_BAG_L1 | 0.53443 |
| LightGBMLarge_BAG_L1 | 1.82482 |

같은 알고리즘이 L1 → L2 에서 크게 개선된다(LightGBMXT −0.058, CatBoost −0.098).
스태킹이 실제로 작동하고 있다는 뜻이고, 그래서 **예측을 못 만드는 것이 더 아깝다.**

## 시도 7 · 8 — 재로드 회피와 해결

**시도 7** `predictor.persist()` 로 메모리 상주 후 같은 셀에서 즉시 예측 →
`persist()` 자체가 모델을 로드하므로 **동일하게 segfault**. `FIT_DONE` 은 찍히고
`PERSIST_DONE` 전에 사망.

**시도 8** `OMP_NUM_THREADS=1` → **해결.**

```bash
OMP_NUM_THREADS=1 AG_TIME_LIMIT=900 AG_REFIT_FULL=0 MPLBACKEND=Agg \
  ../.venv-ag/bin/marimo export html 07_autogluon_stack.py -o out/07_autogluon_stack.html
```

```
FIT_DONE / PERSIST_DONE / PREDICT_DONE (144368, 9) / FITTED=21
SUBMISSION_WRITTEN rows=144368
segfault: 0
```

대가가 있다 — 단일 스레드라 느려진다.

| | 600초 (멀티스레드, 크래시) | 900초 (단일스레드, 성공) |
|---|---|---|
| 최고 val | 0.41685 | 0.42300 |
| bag folds | 8 | 5 |

시간을 1.5배 줬는데 점수는 0.006 나빠졌다. 크래시를 피한 비용이다.

제출 결과는 **Public 0.41182 / Private 0.41415 (68위 / 3,509, 상위 1.9%)**.

# 2. `autogluon.tabular` 만 설치하면 부스팅 백엔드가 없다

```
lightgbm ✗  xgboost ✗  catboost ✗  torch ✗   (sklearn 만 OK)
```

그대로 돌리면 sklearn 모델만 학습하고 **정상 종료**한다.
`autogluon.tabular[all]` 로 설치해야 한다.
새 환경에서는 `fit()` 전에 백엔드 import 를 먼저 테스트할 것.

---

# 3. AutoGluon 1.6.3 × pandas 3.x 비호환

```
TypeError: Cannot interpret '<StringDtype(na_value=nan)>' as a data type
```

AutoGluon 1.6.3 은 pandas 3.x 의 `StringDtype` 을 처리하지 못한다. `pandas<3` 필요.
이때도 모델 0개 학습에 exit code 0 이다.

---

# 4. (사내) Fasoo DRM 이 패키지 메타데이터를 암호화

uv 로 설치하면 `.txt` 메타데이터가 DRM 암호화되어 import 가 깨진다.

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x9b in position 0
파일 내용: "\x9b DRMONE  This Document is encrypted and protected by Fasoo DRM"
```

- 대상: `top_level.txt`, `entry_points.txt`, `VERSION.txt`
- **증상이 원인과 무관해 보인다** — `import torch` 실패의 범인이 pandas 의
  `entry_points.txt` 였다 (`importlib.metadata` 가 전체 dist 를 순회하다 만남)
- **pip 로 설치하면 발생하지 않는다.** uv 의 아카이브 추출이 트리거
- 경로 무관 (`/tmp`, `~/orca`, `~/Desktop` 모두 동일)
- 복구: 정상 파일을 직접 덮어쓰기 (직접 쓴 `.txt` 는 암호화되지 않음)

---

# 교훈

- **exit code 0 을 믿으면 안 된다.** 이번 작업에서 조용한 실패가 3종류 나왔다
  (백엔드 미설치 / pandas 비호환 / segfault). 전부 셸은 성공으로 보고했다
- **최소 재현이 통과해도 가설을 배제하면 안 된다.** libomp 가설을 한 번 버렸는데,
  재현 스크립트의 import 순서가 우연히 안전한 쪽이었을 뿐이다
- 환경 문제를 의심할 때 **검색을 먼저 했어야 했다.** 알려진 이슈였고,
  동료 피드백 한 줄("맥 libomp 충돌 검색해보라")이 6번의 시행착오보다 빨랐다
