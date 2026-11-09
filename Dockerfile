FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

# Copy application source and pre-extracted knowledge base
COPY . .

ENV PORT=8501
EXPOSE 8501

CMD ["sh", "-c", "uv run streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
