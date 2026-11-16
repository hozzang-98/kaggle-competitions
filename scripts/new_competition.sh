#!/usr/bin/env bash
# 새 대회 시작: 템플릿 복사 + 캐글 데이터 다운로드 + 제출 포맷 확인
# usage: ./scripts/new_competition.sh <kaggle-slug>
set -euo pipefail

SLUG="${1:-}"
if [ -z "$SLUG" ]; then
  echo "usage: $0 <kaggle-slug>" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/competitions/$SLUG"

if [ -d "$DEST" ]; then
  echo "이미 존재함: competitions/$SLUG" >&2
  exit 1
fi

# 종료된 대회는 `kaggle competitions list -s <slug>` 에 안 잡힌다.
# slug를 직접 지정하는 아래 명령은 동작한다. 403이면 웹에서 룰 동의가 필요하다.
echo "== 대회 파일 목록 =="
kaggle competitions files -c "$SLUG"

cp -r "$ROOT/competitions/_template" "$DEST"
sed -i '' "s/<대회명>/$SLUG/g; s/<slug>/$SLUG/g" "$DEST/README.md"

kaggle competitions download -c "$SLUG" -p "$DEST/data"
unzip -o "$DEST/data/*.zip" -d "$DEST/data"

echo
echo "== 받은 파일 =="
ls -1 "$DEST/data"

# 제출 포맷은 매번 걸리는 지점이라 자동으로 보여준다.
# sample_submission.csv 가 없고 benchmark 류 파일이 그 역할인 대회도 있다.
echo
echo "== 제출 포맷 후보 =="
FOUND=0
for f in "$DEST"/data/*.csv; do
  base="$(basename "$f")"
  case "$base" in
    *sample*|*submission*|*benchmark*)
      echo "--- $base ($(( $(wc -l < "$f") - 1 ))행) ---"
      head -3 "$f"
      echo
      FOUND=1
      ;;
  esac
done
if [ "$FOUND" -eq 0 ]; then
  echo "sample submission 으로 보이는 파일이 없다. 대회 페이지에서 직접 확인할 것."
fi

echo "완료: competitions/$SLUG"
