FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY src/ ./src/
COPY models/ ./models/
COPY tests/ ./tests/

# Создание non-root пользователя для безопасности
RUN useradd -m -u 1000 mluser && chown -R mluser /app
USER mluser

# Переменные окружения
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=/app/mlruns

# Точка входа
EXPOSE 8000

CMD ["python", "-c", "from src.train import train_models; train_models()"]