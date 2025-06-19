import pandas as pd
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings,  ChatOpenAI
from langchain.agents.agent import AgentExecutor
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from lib.tools import load_csv_as_dataframe, load_data
from tools.stock import ProductStockTool

def initialize_agent(api_key):
    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo",
        max_tokens=1000,
        api_key=api_key
    )
    return llm


def create_agent(api_key: str, df: pd.DataFrame):
    llm = initialize_agent(api_key)
    stock_tool = ProductStockTool(csv_data=df)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are xventory, an AI assistant specialized in inventory management.
            Your primary function is to help users check product stock levels and inventory information.

            Key capabilities:
            - Check stock quantities for products
            - Identify low stock and out-of-stock items
            - Provide detailed inventory information
            - Search products by name, SKU, or description

            Important guidelines:
            - Focus only on inventory-related queries
            - Use the check_product_stock tool for all stock-related questions
            - Provide clear, actionable inventory information
            - If asked about non-inventory topics, politely redirect to inventory matters

            Always be helpful and provide detailed stock information when available."""),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm, [stock_tool], prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[stock_tool],
        verbose=True,
        handle_parsing_errors=True
    )
    return agent_executor


def run_query(agent_executor, query):
    try:
        response = agent_executor.invoke({"input": query})
        return response.get("output", "No response from agent.")
    except Exception as e:
        return f"Error during agent execution: {str(e)}"