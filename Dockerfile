# Smart Grid System Docker Image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs models outputs

# Set environment variables
ENV PYTHONPATH=/app
ENV GRID_ENV=production

# Expose dashboard port
EXPOSE 8050

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "dashboard" ]; then\n\
    python main.py dashboard --host 0.0.0.0 --port 8050\n\
elif [ "$1" = "generate" ]; then\n\
    python main.py generate\n\
elif [ "$1" = "pipeline" ]; then\n\
    python main.py pipeline\n\
elif [ "$1" = "backtest" ]; then\n\
    python main.py backtest\n\
elif [ "$1" = "test" ]; then\n\
    python test_suite.py smoke\n\
else\n\
    python main.py "$@"\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["dashboard"]