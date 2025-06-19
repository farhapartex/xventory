from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
import pandas as pd

def load_data(file_path: str) -> List[Document]:
    loader = CSVLoader(
        file_path=file_path,
        encoding="utf-8",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
        }
    )
    data = loader.load()
    return data

def load_csv_as_dataframe(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        return df
    except Exception as e:
        return pd.DataFrame()

