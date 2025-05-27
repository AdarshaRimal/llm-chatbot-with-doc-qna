from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging

logger = logging.getLogger(__name__)

def build_agent(llm, tools, system_message, memory=None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prompt=prompt,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
