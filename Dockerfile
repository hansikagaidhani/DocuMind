# HuggingFace Spaces Docker SDK
# Runs FastAPI on :8000 (internal) + Streamlit on :7860 (exposed)

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

EXPOSE 7860

ENV API_URL=http://localhost:8000
ENV PYTHONUNBUFFERED=1

CMD ["./start.sh"]
