from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from query_vector_db import rag_chain
from insert_record_vector_db import PdfReader, retriever, text_splitter
from pydantic import BaseModel
from io import BytesIO

# Define the request body schema
class QueryRequest(BaseModel):
    query: str

app = FastAPI()

@app.get('/')
def home_page():
    return {"Message":"Hello, welcome to rag application."}

@app.post('/query')
async def user_query(request: QueryRequest):
    user_request = request.query
    return {"Result": rag_chain.invoke(user_request)}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Read file bytes
    pdf_bytes = await file.read()

    # Wrap bytes in a file-like object
    pdf_stream = BytesIO(pdf_bytes)

    # Load with PyPDF
    reader = PdfReader(pdf_stream)

    # read text from pdf
    raw_text = ''
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    texts = text_splitter.split_text(raw_text)

    retriever.add_texts(texts[:50])

    return JSONResponse({
        "filename": file.filename,
        "size_bytes": len(pdf_bytes),
        "message": "PDF uploaded successfully"
    })
