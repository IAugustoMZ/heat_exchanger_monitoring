FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "starlette<1.0.0" && \
    pip install --no-cache-dir mlflow==3.10.1 scikit-learn==1.8.0 cloudpickle==3.1.2 numpy pandas pyarrow scipy

COPY mlflow/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 5000 5001

ENTRYPOINT ["/app/entrypoint.sh"]
