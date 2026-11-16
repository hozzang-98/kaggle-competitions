#!/usr/bin/env bash
# 새 대회 시작: 템플릿 복사 + 캐글 데이터 다운로드
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

cp -r "$ROOT/competitions/_template" "$DEST"

# README 플레이스홀더를 슬러그로 치환
sed -i '' "s/<대회명>/$SLUG/g; s/<slug>/$SLUG/g" "$DEST/README.md"

kaggle competitions download -c "$SLUG" -p "$DEST/data"
unzip -o "$DEST/data/*.zip" -d "$DEST/data"

echo
echo "완료: competitions/$SLUG"
echo "받은 파일:"
ls -1 "$DEST/data"
