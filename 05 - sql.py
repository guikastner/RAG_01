import os
import authorization
from langchain import hub
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = authorization.apikey

model = ChatOpenAI(model="gpt-4")

db = SQLDatabase.from_uri("sqlite:///ipca.db")

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model,
    )

system_message = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm=model,
    tools=toolkit.get_tools(),
    prompt=system_message
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

prompt = '''
    User as ferramentas para responder perguntas sobre economia e o índice IPCA ao longo do tempo
    Responda em PT-BR
    Perguntas: {q}
'''

prompt_template = PromptTemplate.from_template(prompt)

question = 'Em quais meses teve deflação? quais os valores?'

response = agent_executor.invoke(
    {'input':prompt_template.format(q=question)}
    )

print(response.get('output'))