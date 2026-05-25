FROM python:3.11.5-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    cloudpickle==2.2.1 \
    scipy==1.14.1

# Copy MLflow model artifacts (akan di-mount oleh CI)
COPY mlflow_model /app/model

EXPOSE 8080

# Start MLflow model server
CMD ["mlflow", "models", "serve", \
     "--model-uri", "/app/model", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--no-conda"]
