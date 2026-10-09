FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app
COPY requirements.txt requirements.in pyproject.toml ./
RUN pip install -r requirements.txt
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN pip install --no-deps -e . && useradd -m app && mkdir -p /app/data /mlartifacts && chown -R app /app /mlartifacts
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "retail_mlops.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
