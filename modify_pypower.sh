#!/bin/sh
# 仅当导入行不存在时，才在第 17 行插入；只匹配 case{number}.py

set -eu

cd ./RL4Grid/model_jm/
pypower_dir="$(python -c 'import os, pypower; print(os.path.dirname(pypower.__file__))')"; \
cp -v -- case* TX* SG* WE* "$pypower_dir"/
cd -
cd ./RL4Grid/pglib-opf/
cp -v -- * "$pypower_dir"/
cd -

pypower_dir="$(python -c 'import os, pypower; print(os.path.dirname(pypower.__file__))')"
api="$pypower_dir/api.py"

if [ ! -f "$api" ]; then
  echo "未找到 $api" >&2
  exit 1
fi

ins="$(mktemp)"; : > "$ins"
found_any=0

# 只遍历 case*.py，但用正则二次过滤，仅保留 case[0-9]+.py
for f in "$pypower_dir"/case*.py; do
  [ -e "$f" ] || break
  mod=$(basename "$f" .py)

  # 仅保留 "case" 后全是数字的文件名
  echo "$mod" | grep -Eq '^case[0-9]+(_[A-Za-z0-9_]+)?$' || continue
  [ "$mod" = "__init__" ] && continue

  line="from .${mod} import ${mod}"
  if ! grep -qxF "$line" "$api"; then
    echo "$line" >> "$ins"
    found_any=1
  fi
done

if [ "$found_any" -eq 0 ]; then
  echo "没有需要新增的 case{number} 导入，或已全部存在。"
  rm -f "$ins"
  exit 0
fi

cp -p "$api" "${api}.bak"

headn=16
tmp="$(mktemp)"
awk -v headn="$headn" -v insfile="$ins" '
NR==headn {
  print
  while ((getline line < insfile) > 0) print line
  close(insfile)
  inserted=1
  next
}
{ print }
END {
  if (!inserted) {
    while ((getline line < insfile) > 0) print line
    close(insfile)
  }
}
' "$api" > "$tmp"

mv "$tmp" "$api"
echo "已在 $api 第 17 行插入以下导入（仅限 case{number}）："
cat "$ins"
rm -f "$ins"
