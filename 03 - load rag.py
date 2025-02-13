import os
import authorization
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser as stro
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


os.environ['OPENAI_API_KEY'] = authorization.apikey


pdf_path = 'p53ug_en.pdf'
#pdf_path = 'laptop_manual.pdf'
loader = PyPDFLoader(pdf_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(
    documents=docs,
)

persist_directory = 'db'


embedding = OpenAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=persist_directory,
    collection_name='laptop_manual',
)