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
    print("xventory AI Agent - Inventory Management Assistant")
    print("=" * 50)

    while True:
        user_input = input("\n You: ").strip()
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        elif user_input.lower() == "memory":
            print("Feature is on the way!")
        else:
            result = run_query(agent, user_input)
            print(f"Agent: {result}")

        print("\n" + "=" * 50)

