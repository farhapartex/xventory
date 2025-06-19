import os
from dotenv import load_dotenv
from agents.agent import create_agent, run_query
from lib.tools import load_csv_as_dataframe

load_dotenv()

OPEN_AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
file_path = os.getenv("FILE_PATH")

if __name__ == "__main__":
    df = load_csv_as_dataframe(file_path)
    agent = create_agent(OPENAI_API_KEY, df)
    # TODO: will move those queries to README later
    example_queries = [
        "Check stock for 'Inverse static help-desk'",
        "What products are low in stock? check first 20 product.",
        "Show me inventory for SKU 354275DF",
        "Check stock status for '59DA72CD'",
        "How many units do we have of 'Virtual grid-enabled intranet'?"
    ]
    print("xventory AI Agent - Inventory Management Assistant")
    print("=" * 50)

    for query in example_queries:
        print(f"\nQuery: {query}\n")
        result = run_query(agent, query)
        print(result)
        print("\n" + "=" * 50)

