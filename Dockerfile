# filepath: c:\Users\lucas\Projects\chat-with-pdf\Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy dependency definitions
COPY Pipfile Pipfile.lock /app/

# Install dependencies
RUN pipenv install --system --deploy --ignore-pipfile

# Copy application code
COPY . /app

# Create directories for PDFs and indices
RUN mkdir -p /app/pdfs /app/indices

# Expose port for FastAPI
EXPOSE 8000

# Default command to run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]