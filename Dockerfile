FROM python:3.9-slim

# Avoid .pyc files and make logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Optional runtime dep for scikit-learn wheels that use OpenMP
# (safe on both amd64 and arm64; tiny)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Gunicorn defaults; can be overridden at runtime
ENV WEB_CONCURRENCY=3
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose app port
EXPOSE 8000

# Start Gunicorn (Django wsgi)
CMD ["gunicorn", "Project_phData.wsgi:application", \
     "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "60"]
