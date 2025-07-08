FROM ghcr.io/astral-sh/uv:0.7.6-python3.11-bookworm-slim

WORKDIR /app

COPY pyproject.toml .

RUN uv sync --no-dev

COPY src/ ./src/
COPY params.yaml ./params.yaml

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--port", "8000", "--host", "0.0.0.0"]
