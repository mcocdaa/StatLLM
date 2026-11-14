FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Install python dependencies (override PIP_INDEX_URL for regional mirrors)
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url ${PIP_INDEX_URL} -r requirements.txt

# Copy application files and seed database
COPY README.md .
COPY pyproject.toml .
COPY statllm/ ./statllm/
COPY web/ ./web/
COPY statllm.db .

# Install statllm using local setuptools without build isolation
RUN pip install --no-cache-dir --no-build-isolation --no-deps -e .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn web.app:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-2}"]
