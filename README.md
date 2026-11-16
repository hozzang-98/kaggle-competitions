# Kaggle

캐글 대회 참가 기록. 대회 하나당 `competitions/` 아래 폴더 하나.

## 대회 목록

| 대회 | 기간 | 태스크 | 주요 접근법 | 순위 / 점수 |
|------|------|--------|-------------|-------------|
| _(첫 대회 추가 예정)_ | | | | |

## 구조

```
common/                 # 대회 공통 유틸 (CV, 메트릭, 시드 고정)
scripts/
  new_competition.sh    # 새 대회 디렉토리 생성 + 데이터 다운로드
competitions/
  _template/            # 새 대회 시작할 때 복사해서 쓰는 템플릿
  <대회명>/
    README.md           # 대회 개요 / 접근법 / 결과 회고
    data/               # gitignore - 캐글 API로 다운로드
    notebooks/          # EDA, 실험
    src/                # 전처리 / 학습 / 추론 스크립트
    submissions/        # 제출 이력
```

## 시작하기

```bash
# 1. 환경 설치
pip install -r requirements.txt

# 2. 노트북 output 자동 제거 훅 설치 (레포 비대화 방지)
nbstripout --install

# 3. 캐글 API 토큰 설정 (최초 1회)
#    kaggle.com > Account > Create New API Token -> kaggle.json 다운로드
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 4. 새 대회 시작 (템플릿 복사 + 데이터 다운로드)
./scripts/new_competition.sh <kaggle-slug>
```

슬러그는 대회 URL 뒤쪽. `kaggle.com/competitions/playground-series-s5e11` -> `playground-series-s5e11`

## 제출

```bash
kaggle competitions submit -c <대회명> -f submissions/<파일>.csv -m "설명"
```

## 참고 문서

캐글 사용법 노트는 옵시디언 볼트에 있음 -> `~/Desktop/git/kaggle-competitions/docs/`

- `Kaggle-제출-가이드.md` — 대회 유형, Code competition 제약, kernel-metadata 전체 설명
- `Titanic-시작하기.md` — 등록부터 첫 제출까지 (실습 가이드)
