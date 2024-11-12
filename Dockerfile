FROM python:3.10-slim

WORKDIR /app

# Combine RUN instructions and address warnings
RUN pip install --no-cache-dir poetry==1.8.4 && \
    apt-get update && \
    apt-get install --yes --no-install-recommends gcc g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy 
COPY common /app/common
COPY ui /app/ui
COPY backend /app/backend
COPY credential /app/credential

ENV GOOGLE_APPLICATION_CREDENTIALS="/app/credential/ai-sandbox-company-73-2659f4150720.json"

# Expose the port your FastAPI app will run on 
EXPOSE 8080

# Command to run the FastAPI app 
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8080"]