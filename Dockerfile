FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for caching
COPY requirements.txt .
# First upgrade pip, then install your requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Start the server (Person A's main.py)
CMD ["python", "main.py"]