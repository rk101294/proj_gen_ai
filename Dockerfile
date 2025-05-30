# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables to reduce image size and improve performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Set working directory
WORKDIR /app

# Install OS-level dependencies (only essential ones)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose Cloud Run port (should be 8080)
EXPOSE 8080

# Start Streamlit app using the PORT env from Cloud Run
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0



