FROM python:3.11-slim

WORKDIR /app

# System deps needed by some ML/plotting libs (e.g. xgboost, shap, matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
    "--server.port=8501", "--server.address=0.0.0.0"]
