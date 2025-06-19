import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings,  ChatOpenAI
from langchain.agents.agent import AgentExecutor
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from tools.stock import VectorProductStockTool


def create_product_documents(df: pd.DataFrame):
    documents = []

    for idx, row in df.iterrows():
        content_parts = []
        content_parts.append(f"Name: {row.get('name', 'N/A')}")
        content_parts.append(f"SKU: {row.get('sku', 'N/A')}")
        content_parts.append(f"Brand: {row.get('brand', 'N/A')}")
        content_parts.append(f"Description: {row.get('description', 'N/A')}")
        content_parts.append(f"Short Description: {row.get('short_description', 'N/A')}")
        content_parts.append(f"Category ID: {row.get('category_id', 'N/A')}")
        content_parts.append(f"Quantity: {row.get('quantity', 0)}")
        content_parts.append(f"Stock Status: {row.get('stock_status', 'unknown')}")
        content_parts.append(f"Low Stock Threshold: {row.get('low_stock_threshold', 10)}")
        content_parts.append(f"Price: {row.get('price', 0)}")
        content_parts.append(f"Cost: {row.get('cost', 0)}")
        content_parts.append(f"Currency: {row.get('currency', 'USD')}")
        content_parts.append(f"Material: {row.get('material', 'N/A')}")
        content_parts.append(f"Model: {row.get('model', 'N/A')}")
        content_parts.append(f"Colors: {row.get('colors', 'N/A')}")
        content_parts.append(f"Sizes: {row.get('sizes', 'N/A')}")
        content_parts.append(f"Weight: {row.get('weight', 'N/A')}")
        content_parts.append(f"Supplier ID: {row.get('supplier_id', 'N/A')}")
        content_parts.append(f"Supplier SKU: {row.get('supplier_sku', 'N/A')}")
        content_parts.append(f"Lead Time: {row.get('lead_time', 'N/A')}")
        content_parts.append(f"SEO Title: {row.get('seo_title', 'N/A')}")
        content_parts.append(f"Tags: {row.get('tags', 'N/A')}")
        content_parts.append(f"Barcode: {row.get('barcode', 'N/A')}")

        content = '\n'.join(content_parts)
        metadata = {
            'product_id': row.get('id', idx),
            'sku': row.get('sku', ''),
            'name': row.get('name', ''),
            'brand': row.get('brand', ''),
            'category_id': row.get('category_id', ''),
            'stock_status': row.get('stock_status', ''),
            'quantity': row.get('quantity', 0),
            'price': row.get('price', 0),
            'row_index': idx
        }
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        documents.append(doc)

    return documents

def setup_vectorstore(
        df: pd.DataFrame,
        embeddings: OpenAIEmbeddings
):
    try:
        documents = create_product_documents(df)
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./xventory_vectorstore"
        )
        return vectorstore
    except Exception as e:
        return None

def initialize_agent(api_key):
    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo",
        max_tokens=1000,
        api_key=api_key
    )
    return llm


def create_agent(api_key, df: pd.DataFrame):
    llm = initialize_agent(api_key)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = setup_vectorstore(df, embeddings)
    stock_tool = VectorProductStockTool(csv_data=df, vectorstore=vectorstore)

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