# Reproducible ExpertRuleFit via Containers

**Goal:** same data + same config ⇒ same rules + same predictions.

A container must define a **deterministic execution envelope**: exact OS + exact Python + exact dependency wheels + single-threaded numerics + fixed seeds + fixed schema. Here's the concrete way to do it.

---

## 1. Pin the Container Base Image by Digest

Tag pinning (`python:3.11-slim`) is not enough — tags move. Use a digest:

```dockerfile
# Dockerfile
FROM python:3.11-slim@sha256:<PINNED_DIGEST>
```

How to get the digest:

```bash
docker pull python:3.11-slim
docker image inspect python:3.11-slim --format '{{index .RepoDigests 0}}'
```

This removes **base image drift** as a source of non-reproducibility.

---

## 2. Lock Python Dependencies Exactly (Ideally with Hashes)

You want "same wheels every time".

### Option A — `requirements.txt` with hashes (best)

Generate with [pip-tools](https://github.com/jazzband/pip-tools) (or [uv](https://github.com/astral-sh/uv)), then install with hashes enforced.

**Generate** (outside container):

```bash
pip-compile --generate-hashes --output-file requirements.lock.txt pyproject.toml
```

**Install** (inside container):

```bash
pip install --require-hashes -r requirements.lock.txt
```

This prevents **dependency drift** and also prevents pip from silently choosing different builds.

### Option B — Poetry lock / uv lock

If you use Poetry or uv, copy in the lockfile and install strictly from it (no resolver at build time).

---

## 3. Force Deterministic Numerics (Single-Thread BLAS/OpenMP)

Even inside a container, multi-threaded BLAS can change floating-point reduction order. Set these in the image:

```dockerfile
ENV PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MKL_DYNAMIC=FALSE
```

This matters if you're using OpenBLAS / Intel MKL through NumPy / SciPy / scikit-learn.

---

## 4. Make the Training Run Deterministic Inside the Container

Inside your training script (or library), enforce:

- **Fixed seeds** — `random`, `numpy`, and any model `random_state`
- **`n_jobs=1`** anywhere you can (CV solvers, EBM training, etc.)
- **Stable feature order** and schema checks

Even better: wrap `fit` in `threadpoolctl`:

```python
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=1):
    model.fit(X, y)
```

That gives you a hard guarantee even if the runtime tries to use more threads.

---

## 5. Freeze the Data Semantics: Schema + Feature Order

A container can't guarantee you fed the same columns in the same order unless you enforce it.

**Best practice:**

- Require a DataFrame input
- Persist `feature_names_` at fit time
- At predict time, reindex: `X = X[feature_names_]`
- Hard-fail if missing/extra columns

This is often the **#1 source** of "same data (kind of) but different predictions".

---

## 6. Produce an Auditable Fingerprint and Verify It In-Container

This upgrades "we think it's reproducible" to "we can **prove** it".

After training, compute SHA-256 over:

- Sorted rules / rule specs
- Coefficients / intercept
- `feature_names_` (ordered)
- Versions (`python`, `numpy`, `sklearn`, etc.)
- Container image digest (record it)

Then add a **reproducibility test** that runs training twice and asserts identical fingerprints.

---

## 7. Practical Docker Setup

### Dockerfile (template)

```dockerfile
FROM python:3.11-slim@sha256:<PINNED_DIGEST>

ENV PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MKL_DYNAMIC=FALSE \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (keep minimal; pin if you need determinism across rebuilds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.lock.txt /app/requirements.lock.txt
RUN pip install --upgrade pip \
 && pip install --require-hashes -r requirements.lock.txt

COPY . /app

# Optional: run tests in image build
RUN pytest -q

ENTRYPOINT ["python", "-m", "your_training_module"]
```

### Build and run

```bash
docker build --pull --no-cache -t expertrulefit-repro .
docker run --rm -v "$PWD/data:/data" expertrulefit-repro
```

> **Cross-machine guarantees:** also pin the platform:
>
> ```bash
> docker build --platform=linux/amd64 ...
> docker run  --platform=linux/amd64 ...
> ```

---

## 8. What Containers Can't Fully Guarantee

Containers greatly reduce variance, but a truly strict guarantee may still require:

| Condition | Why it matters |
|---|---|
| Same CPU architecture (amd64 vs arm64) | Different FP rounding on different ISAs |
| Pinned base image digest | Tag-only pinning drifts silently |
| Pinned wheels with hashes | Prevents pip from choosing different builds |
| Single-threaded numerics | Multi-threaded BLAS breaks FP associativity |
| Deterministic code paths | Unordered `set`/`dict` iteration can change feature order |
| No epsilon-scale selection thresholds | Tiny numeric differences can flip rule inclusion |

---

## The Guarantee

If you implement all of the above, you can responsibly say:

> **Guaranteed reproducible within a deterministic execution envelope: pinned container digest, pinned dependency wheels, single-threaded numerics, fixed seeds, and validated schema/feature order ⇒ identical rules and identical predictions.**
