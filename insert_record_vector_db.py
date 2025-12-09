from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pinecone import Pinecone,ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
load_dotenv()

## vector embedding and sparse matrix
hf_token=os.getenv('HF_TOKEN')
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"token": hf_token})

index_name="hybrid-search-pinecone-pdf-document"
## initialize the Pinecone client
pc=Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

#create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # vector dimension
        metric="dotproduct",  # dotproduct supports sparse vector
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index=pc.Index(index_name)

bm25_encoder=BM25Encoder().default()

pdfreader = PdfReader(r'data/budget_speech.pdf')
# read text from pdf
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

## tfidf values on these sentence
bm25_encoder.fit(texts[:50])

## store the values to a json file
bm25_encoder.dump("bm25_values.json")

# load to your BM25Encoder object
bm25_encoder = BM25Encoder().load("bm25_values.json")

retriever=PineconeHybridSearchRetriever(embeddings=embeddings,
                                        sparse_encoder=bm25_encoder,
                                        index=index,
                                        top_k=2 )

retriever.add_texts( texts[:50])