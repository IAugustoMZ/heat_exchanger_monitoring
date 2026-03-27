#!/bin/bash
set -e

echo "Starting MLflow Tracking Server on port 5000..."
mlflow server \
    --backend-store-uri sqlite:///app/mlflow.db \
    --default-artifact-root /app/mlartifacts \
    --host 0.0.0.0 \
    --port 5000 &

# Wait for tracking server to be ready
echo "Waiting for MLflow Tracking Server..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:5000/health > /dev/null 2>&1; then
        echo "MLflow Tracking Server is ready."
        break
    fi
    sleep 2
done

MODEL_PATH="/app/mlartifacts/1/models/m-b10b0d77d05344c79c6dd1eb756c8ef7/artifacts"
echo "Starting MLflow Model Serving on port 5001..."
echo "Serving model from: $MODEL_PATH"

mlflow models serve \
    --model-uri "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 5001 \
    --env-manager local \
    --no-conda &

# Wait for model serving to be ready
echo "Waiting for MLflow Model Serving..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:5001/health > /dev/null 2>&1; then
        echo "MLflow Model Serving is ready."
        break
    fi
    sleep 3
done

echo "All MLflow services started successfully."

# Keep container alive
wait
