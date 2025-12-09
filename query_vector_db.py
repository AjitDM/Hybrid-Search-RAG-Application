from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

## vector embedding and sparse matrix
hf_token=os.getenv('HF_TOKEN')
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"token": hf_token})

index_name="hybrid-search-pinecone-pdf-document"
## initialize the Pinecone client
pc=Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index=pc.Index(index_name)

# load to your BM25Encoder object
bm25_encoder = BM25Encoder().load(r"artifacts\bm25_values.json")

retriever=PineconeHybridSearchRetriever(embeddings=embeddings,
                                        sparse_encoder=bm25_encoder,
                                        index=index,
                                        top_k=2 )

groq_api_key=os.getenv('GROQ_API_KEY')
llm=ChatGroq(groq_api_key=groq_api_key,model_name="qwen/qwen3-32b")

# Define the RAG prompt template
rag_template = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
    ("user", "Question: {question}\nContext: {context}")
])

# Construct the RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # Retrieve context based on the question
    | rag_template  # Apply the prompt template
    | llm  # Pass to the LLM
    | StrOutputParser()  # Parse the output as a string
)

__all__=['rag_chain']

