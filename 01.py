import os
import authorization
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


os.environ['OPENAI_API_KEY'] = authorization.apikey

model = ChatOpenAI(
    model="gpt-4"
    )

pdf_path = 'p53ug_en.pdf'
#pdf_path = 'laptop_manual.pdf'
loader = PyPDFLoader(pdf_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = text_splitter.split_documents(
    documents=docs
)

#print(chunks)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name='laptop_manual'
    )

retriever_vector = vector_store.as_retriever()

result =retriever_vector.invoke(
    'qual é a bateria do laptop?',
)

print(result)