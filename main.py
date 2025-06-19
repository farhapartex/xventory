import os
from typing import Union
from fastapi import FastAPI
from dotenv import load_dotenv
from models.query import QueryRequest
from agents.agent import create_agent, run_query
from lib.tools import load_csv_as_dataframe

load_dotenv()

OPEN_AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
file_path = os.getenv("FILE_PATH")

app = FastAPI(
        title="xventory AI Agent",
        description="An AI agent for inventory management",
        version="1.0.0"
    )
df = load_csv_as_dataframe(file_path)
agent = create_agent(OPENAI_API_KEY, df)

@app.get("/api/v1/health-check")
async def health_check() -> Union[str, dict]:
    try:
        return {"response": "Working"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/v1/ask-agent")
async def run_agent(request: QueryRequest) -> Union[str, dict]:

    try:
        result = run_query(agent, request.query)
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}


