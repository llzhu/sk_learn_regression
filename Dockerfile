FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libsm6 \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

    
# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Send logs directly to terminal
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "ML_Regression_Home.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
