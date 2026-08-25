#!/usr/bin/env bash
# Retry `lms get` on a HuggingFace URL until the download completes.
LMS="C:/Users/<you>/.lmstudio/bin/lms.exe"
URL="$1"; NAME="$2"; MAX="${3:-20}"
for i in $(seq 1 "$MAX"); do
  if "$LMS" ls --json < /dev/null 2>/dev/null | grep -qi "$NAME"; then
    echo "$(date '+%H:%M:%S') $NAME present after $((i-1)) attempt(s)"; exit 0
  fi
  echo "$(date '+%H:%M:%S') attempt $i: $URL"
  "$LMS" get "$URL" -y < /dev/null >> "logs/get_${NAME}.log" 2>&1
  sleep 5
done
echo "$(date '+%H:%M:%S') gave up on $NAME"; exit 1
