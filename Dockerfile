# Optional container for the online scoring service (Docker is not required by the brief).
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY serving/ serving/
COPY artifacts/ artifacts/
EXPOSE 8000
HEALTHCHECK CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
