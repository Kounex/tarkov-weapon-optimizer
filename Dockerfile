# Use an official Python slim image as a base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# Some python packages like ortools might need libgomp1 or other libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories for the app
RUN mkdir -p logs .cache && chmod 777 logs .cache

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE ${STREAMLIT_SERVER_PORT}

# Add a healthcheck using the configured port
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:${STREAMLIT_SERVER_PORT}/_stcore/health || exit 1

# Command to run the application
# Streamlit honors STREAMLIT_SERVER_PORT and STREAMLIT_SERVER_ADDRESS env vars automatically
ENTRYPOINT ["streamlit", "run", "app.py"]
