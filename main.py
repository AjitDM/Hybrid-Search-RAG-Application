from fastapi import FastAPI
from query_vector_db import rag_chain
from pydantic import BaseModel

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
