# 1.1 - Base Python 3.11-slim image
FROM python:3.11-slim

# 1.2 - Set environment variables and working directory
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 1.2 - Install uv package manager
RUN pip install uv
# 1.3 - Copy dependency files and install them
COPY pyproject.toml uv.lock ./
ENV UV_HTTP_TIMEOUT=300
RUN uv sync --frozen --no-dev

# 1.4 - Copy source code
COPY . .

# 1.5 - Set PYTHONPATH so imports work correctly
ENV PYTHONPATH=/app:/app/app

# 1.6 - Configure port and startup command
EXPOSE 8000

CMD ["uv", "run", "python", "app/main.py"]
