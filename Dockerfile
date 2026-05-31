FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies needed for ctranslate2 and pandas
RUN apt-get update \
    && apt-get install -y --no-install-recommends libomp-dev build-essential elfutils patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python - <<'PY'
import importlib.util
import pathlib
import subprocess
import sys

spec = importlib.util.find_spec("ctranslate2")
if not spec or not spec.origin:
    print("ctranslate2 package not found; skipping execstack fix", file=sys.stderr)
    raise SystemExit(0)

pkg_dir = pathlib.Path(spec.origin).parent
print(f"Clearing execstack flag under {pkg_dir}", file=sys.stderr)
found_any = False
for so_path in pkg_dir.rglob("*.so"):
    found_any = True
    result = subprocess.run(["patchelf", "--set-execstack", "0", str(so_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"patchelf failed for {so_path}: {result.stderr}", file=sys.stderr)
    else:
        print(f"Updated execstack flag for {so_path}", file=sys.stderr)

if not found_any:
    print("No shared libraries found under ctranslate2; nothing to patch", file=sys.stderr)
PY

COPY . .

EXPOSE 3415 3417

ENV APP_PORT=3415 \
    CTRANSLATE_DEVICE=cpu \
    CTRANSLATE_DEVICE_INDEX=0

CMD ["sh", "-c", "uvicorn run_endpoint:app --host 0.0.0.0 --port ${APP_PORT:-3415} ${UVICORN_RELOAD:+--reload}"]
