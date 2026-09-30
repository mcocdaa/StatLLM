FROM python:3.12-slim

WORKDIR /app


# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY statllm/ ./statllm/
COPY web/ ./web/
COPY pyproject.toml .

# Install statllm package
RUN pip install --no-cache-dir -e .

# Expose web port
EXPOSE 8000

# Run uvicorn
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
