FROM python:3.10-slim

# Create a non-root user for HF Spaces security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files with correct ownership
COPY --chown=user . .

# Expose the HF default port
EXPOSE 7860

# Start the server
CMD ["python", "main.py"]