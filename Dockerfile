FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

# Copy application source and pre-extracted knowledge base
COPY . .

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
