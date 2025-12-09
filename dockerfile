FROM python:3.10-slim
WORKDIR /hybrid-search-rag-api

RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY main.py /hybrid-search-rag-api/main.py
COPY query_vector_db.py /hybrid-search-rag-api/query_vector_db.py
COPY insert_record_vector_db.py /hybrid-search-rag-api/insert_record_vector_db.py
COPY requirement_docker.txt /hybrid-search-rag-api/requirement_docker.txt
COPY /artifacts /hybrid-search-rag-api/artifacts

RUN pip3 install --no-cache-dir -r requirement_docker.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
