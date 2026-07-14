FROM python:3.11-slim

WORKDIR /app

# Copy project first (includes source + config)
COPY . .

# Install the project itself (not a local venv)
RUN pip install --no-cache-dir -e .

# Default: fake executor CPU benchmark
CMD python examples/benchmark.py \
    --executor fake \
    --requests 32 \
    --tokens 16
