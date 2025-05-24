#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status

# Model name can be passed as an environment variable OLLAMA_MODEL_TO_PULL
# Default to llama3.2 if not set
MODEL_NAME=${OLLAMA_MODEL_TO_PULL:-llama3.2}

echo "Ollama Initialization Script: Starting..."
echo "Target model: ${MODEL_NAME}"

# Install curl if not available (common in minimal Docker images)
if ! command -v curl > /dev/null 2>&1; then
    echo "curl not found, installing..."
    apk add --no-cache curl || apt-get update && apt-get install -y curl || {
        echo "Failed to install curl. Trying with wget as fallback..."
        if ! command -v wget > /dev/null 2>&1; then
            apk add --no-cache wget || apt-get install -y wget || {
                echo "ERROR: Neither curl nor wget could be installed. Cannot proceed with health checks."
                exit 1
            }
        fi
        USE_WGET=true
    }
fi

# Function to check if Ollama is responsive
check_ollama_health() {
    if [ "${USE_WGET:-false}" = "true" ]; then
        wget -q --spider "http://localhost:11434/" 2>/dev/null
    else
        curl -s -f "http://localhost:11434/" > /dev/null 2>&1
    fi
}

# Start ollama serve in the background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!
echo "Ollama server process started in background with PID ${OLLAMA_PID}."

# Wait for Ollama server to be responsive
echo "Waiting for Ollama server to become responsive..."
max_retries=60  # Increased timeout for slower systems
count=0
until check_ollama_health; do
    if [ ${count} -ge ${max_retries} ]; then
        echo "Ollama server did not become responsive after ${max_retries} retries ($(($max_retries * 2)) seconds). Exiting."
        kill ${OLLAMA_PID} 2>/dev/null || true
        wait ${OLLAMA_PID} 2>/dev/null || true
        exit 1
    fi
    echo "Ollama not responsive yet (attempt $((count+1))/${max_retries}). Waiting 2 seconds..."
    sleep 2
    count=$((count+1))
done
echo "Ollama server is responsive."

# Check if the model is already downloaded
echo "Checking if model '${MODEL_NAME}' is already available..."
if ollama list | grep -q "^${MODEL_NAME}"; then
    echo "Model '${MODEL_NAME}' is already available locally. No pull needed."
else
    echo "Model '${MODEL_NAME}' not found locally. Pulling model..."
    echo "This may take several minutes depending on model size and network speed..."
    
    # Pull model with timeout and retry logic
    PULL_RETRIES=3
    PULL_COUNT=0
    
    while [ ${PULL_COUNT} -lt ${PULL_RETRIES} ]; do
        echo "Attempting to pull model (attempt $((PULL_COUNT+1))/${PULL_RETRIES})..."
        
        if ollama pull "${MODEL_NAME}"; then
            echo "Model '${MODEL_NAME}' pulled successfully."
            break
        else
            PULL_COUNT=$((PULL_COUNT+1))
            if [ ${PULL_COUNT} -lt ${PULL_RETRIES} ]; then
                echo "Pull failed. Retrying in 10 seconds..."
                sleep 10
            else
                echo "ERROR: Failed to pull model '${MODEL_NAME}' after ${PULL_RETRIES} attempts."
                echo "Please check:"
                echo "  - Model name is correct (try: ollama search ${MODEL_NAME})"
                echo "  - Network connectivity"
                echo "  - Available disk space"
                echo "  - Ollama service status"
                echo ""
                echo "Available models can be found at: https://ollama.ai/library"
                echo "Continuing with Ollama server running, but model may not be available..."
                break
            fi
        fi
    done
fi

# Verify model is available after pull
echo "Verifying model availability..."
if ollama list | grep -q "^${MODEL_NAME}"; then
    echo "✓ Model '${MODEL_NAME}' is ready for use."
else
    echo "⚠ Warning: Model '${MODEL_NAME}' may not be available. Check logs above."
fi

echo ""
echo "=== Ollama Initialization Complete ==="
echo "Server PID: ${OLLAMA_PID}"
echo "Target Model: ${MODEL_NAME}"
echo "Health Check URL: http://localhost:11434/"
echo "=========================================="
echo ""

# Setup signal handlers for graceful shutdown
trap 'echo "Received signal, shutting down Ollama server..."; kill ${OLLAMA_PID} 2>/dev/null || true; wait ${OLLAMA_PID} 2>/dev/null || true; exit 0' INT TERM

# Wait for the ollama serve process to exit. This keeps the container alive.
# If ollama serve exits for any reason, the script (and thus the container) will exit.
wait ${OLLAMA_PID}