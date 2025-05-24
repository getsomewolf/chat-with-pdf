#!/bin/sh
set -e # Exit immediately if a command exits with a non-zero status

# Model name can be passed as an environment variable OLLAMA_MODEL_TO_PULL
# Default to llama3.2 if not set
MODEL_NAME=${OLLAMA_MODEL_TO_PULL:-llama3.2}

echo "Ollama Initialization Script: Starting..."
echo "Target model: ${MODEL_NAME}"

# Start ollama serve in the background
ollama serve &
OLLAMA_PID=$!
echo "Ollama server process started in background with PID ${OLLAMA_PID}."

# Wait for Ollama server to be responsive
echo "Waiting for Ollama server to become responsive..."
max_retries=30
count=0
until curl -s -f "http://localhost:11434/" > /dev/null; do
    if [ ${count} -ge ${max_retries} ]; then
        echo "Ollama server did not become responsive after ${max_retries} retries. Exiting."
        kill ${OLLAMA_PID} # Attempt to kill the background server
        wait ${OLLAMA_PID} 2>/dev/null
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
    ollama pull "${MODEL_NAME}"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to pull model '${MODEL_NAME}'. Please check the model name and network."
        # Decide if you want to exit or let ollama serve continue
        # exit 1 
    else
        echo "Model '${MODEL_NAME}' pulled successfully."
    fi
fi

echo "Ollama initialization complete. Ollama server (PID ${OLLAMA_PID}) continues to run."
# Wait for the ollama serve process to exit. This keeps the container alive.
# If ollama serve exits for any reason, the script (and thus the container) will exit.
wait ${OLLAMA_PID}