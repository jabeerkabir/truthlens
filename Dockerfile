FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only FIRST (much smaller than default)
RUN pip install --no-cache-dir \
    torch==2.2.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install everything else
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV MODEL_ID=Jabirkabir/truthlens-fakenews-detector
ENV HF_TOKEN=""
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
