# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and MLflow models
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Command to run FastAPI using uvicorn
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
