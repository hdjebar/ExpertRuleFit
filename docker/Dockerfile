# Pinned research stack for ExpertRuleFit experiments and tests.
# Base image: python:3.11-slim, pinned by digest (pulled 2026-11-22 via
# mirror.gcr.io, the Docker Hub mirror; same content-addressed digest).
# Override the registry (not the digest) if Docker Hub is unreachable, e.g.
#   --build-arg BASE_IMAGE=mirror.gcr.io/library/python@sha256:07d48240...
ARG BASE_IMAGE=python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535
FROM ${BASE_IMAGE}

# Single-threaded numerics: part of the determinism envelope.
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY docker/requirements.lock.txt /app/requirements.lock.txt
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --require-hashes --no-deps -r /app/requirements.lock.txt

COPY pyproject.toml README.md /app/
COPY expertrulefit /app/expertrulefit
RUN pip install --no-deps --no-build-isolation -e /app

# Tests, benchmark and experiments are mounted or copied at run time:
#   docker run --rm -v "$PWD":/app -w /app erf-pinned pytest -q
CMD ["python", "-c", "import expertrulefit, sklearn, numpy; print(expertrulefit.__version__, sklearn.__version__, numpy.__version__)"]
