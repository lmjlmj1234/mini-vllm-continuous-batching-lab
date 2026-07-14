# Docker Deployment

> **Current version: Fake Executor CPU demo.**
> This configuration runs the built-in `FakeModelExecutor` — no GPU, no real model
> inference. It validates the Docker build, project install, and benchmark
> pipeline inside a container.
>
> Metrics produced (latency, throughput) are **simulation numbers** and DO NOT
> represent real GPU inference performance.

---

## File roles

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.11-slim image; copies project source; `pip install -e .`; default CMD runs the fake-executor benchmark. |
| `compose.yaml` | Orchestrates the build and run; binds `./benchmark_results` to `/app/benchmark_results` so container output persists locally. |
| `.dockerignore` | Excludes git metadata, editor configs, Python caches, local venvs, model weight files and build artifacts from the Docker build context. |

---

## Commands

### 1. Validate compose file

```bash
docker compose config
```

### 2. Build image

```bash
docker compose build
```

Or tag manually:

```bash
docker build -t mini-vllm-lab:latest .
```

### 3. Run benchmark

```bash
# Use compose (recommended)
docker compose run --rm mini-vllm

# Or vanilla docker (same result)
docker run --rm mini-vllm-lab:latest
```

### 4. Override benchmark parameters

Every CLI flag supported by `examples/benchmark.py` can be overridden.

```bash
# Fewer requests, more tokens, save JSON output
docker compose run --rm mini-vllm \
  python examples/benchmark.py \
    --executor fake \
    --requests 8 \
    --tokens 32 \
    --json-output benchmark_results/results.json

# Quiet mode, custom block size
docker compose run --rm mini-vllm \
  python examples/benchmark.py \
    --executor fake \
    --requests 4 \
    --quiet \
    --block-size 8

# See all available options
docker compose run --rm mini-vllm \
  python examples/benchmark.py --help
```

### 5. View logs

```bash
# Re-attach if running with `docker compose up` (non-detached)
# Or simply check the output from `docker compose run` above.
# For detached mode:
docker compose up -d
docker compose logs -f
```

### 6. Stop and clean up

```bash
docker compose down
```

Remove the image explicitly:

```bash
docker rmi mini-vllm-lab:latest
```

---

## Benchmark results persistence

When you run with `--json-output`, pass a path under `/app/benchmark_results/` to
have the file written to the host's `./benchmark_results/` directory:

```bash
mkdir -p benchmark_results
docker compose run --rm mini-vllm \
  python examples/benchmark.py \
    --executor fake --requests 4 \
    --json-output /app/benchmark_results/my-run.json
```

---

## Next steps — Qwen GPU Docker

To add a GPU-enabled variant:

1. Add `nvidia/cuda:12.4.0-runtime-ubuntu22.04` or
   `nvidia/cuda:12.4.0-devel-ubuntu22.04` as the base image.
2. Install torch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu124`
3. Install `transformers>=4.40` and `accelerate>=0.30`.
4. Add `deploy` / `resources` in compose.yaml:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```
5. Run with `--executor qwen`.
