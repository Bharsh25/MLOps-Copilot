FROM python:3.12-slim
WORKDIR /app
COPY api/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 -r requirements.txt
COPY api/main.py api/model.joblib api/model_meta.json ./
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]