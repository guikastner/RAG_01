import os
import authorization
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser as stro
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


os.environ['OPENAI_API_KEY'] = authorization.apikey

model = ChatOpenAI(
    model="gpt-4"
    )

persist_directory = 'db'
embedding = OpenAIEmbeddings()

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
    collection_name='laptop_manual',
)

retriever = vector_store.as_retriever()

system_prompt = '''
Use o contexto para responder as perguntas.
Contexto: {context}
'''

prompt = ChatPromptTemplate.from_template(
    [
        ('system', system_prompt),
        ('human', '{input}'),
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm = model,
    prompt=prompt,
)

chain = create_retrieval_chain(
    retriever=retriever,
    combine_documents_chain=question_answer_chain
)

query = 'qual a marca e modelo do notebook?'
responde = chain.invoke(
    {
        'input': query,
    }
)

print(responde)