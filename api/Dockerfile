FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ALLOWED_ORIGINS="*"

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy minimal application code, serving models, and demo samples
COPY insightx/__init__.py ./insightx/__init__.py
COPY api/ ./api/
COPY models/ ./models/
COPY samples/ ./samples/

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
