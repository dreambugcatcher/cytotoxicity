FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libsm6 \
    libice6 \
    libfontconfig1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt


COPY ./src/frontend ./src/frontend
COPY ./src/models/models_engine/model_engine.pkl ./src/models/models_engine/model_engine.pkl

CMD ["python", "./src/frontend/main.py"]