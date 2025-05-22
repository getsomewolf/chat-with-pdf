# Dockerfile for FastAPI application with Pipenv
FROM python:3.10-slim

# Force CPU-only mode for ML libraries
# Uncomment the following lines if you want to run on CPU only
ENV CUDA_VISIBLE_DEVICES=""
ENV TOKENIZERS_PARALLELISM="false"
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy dependency definitions
COPY Pipfile Pipfile.lock /app/

# Install dependencies
# Using --system to install into the system Python, common for Docker
# Using --deploy for production to ensure Pipfile.lock is up-to-date and fail if not
# --ignore-pipfile is used if you only want to install from Pipfile.lock
RUN pipenv install --system --deploy --ignore-pipfile

# Copy application source code
COPY ./src /app/src
COPY ./.env.example /app/.env.example 
# If you have other top-level files like main cli.py or api.py directly in root, copy them too
# For this project, main entry points are in src/
# COPY cli.py /app/ # If cli.py was in root
# COPY api.py /app/ # If api.py was in root (now src/api/main.py)

# Expose port for FastAPI
EXPOSE 8000

# Default command to run the API
# Ensure this path matches your FastAPI app instance location
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]