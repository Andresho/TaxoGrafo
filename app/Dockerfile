# Dockerfile for the Pipeline API server
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install runtime dependencies
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY app /app

# Expose port
EXPOSE 8000

# Launch the API server after running database migrations
CMD ["bash", "-c", "alembic upgrade head && uvicorn main:app --host 0.0.0.0 --port 8000"]