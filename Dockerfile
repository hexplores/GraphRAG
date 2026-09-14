FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src HOST=0.0.0.0 PORT=8000

EXPOSE 8000

CMD ["uvicorn", "graphrag_smart_retrieval.api:app", "--host", "0.0.0.0", "--port", "8000"]
