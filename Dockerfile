FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY copilot/ ./copilot/
COPY scripts/ ./scripts/
COPY dashboard/ ./dashboard/
COPY setup.py .
COPY audit_config.yaml .

# Install the package
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/data /app/reports /app/batch_reports

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AUDIT_LOGGING_CONSOLE=true

# Default command (can be overridden)
CMD ["python", "-m", "scripts.audit_runner", "--help"]