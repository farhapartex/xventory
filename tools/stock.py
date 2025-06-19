import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class StockCheckInput(BaseModel):
    query: str

class ProductStockTool(BaseTool):
    name: str = "check_product_stock"
    description: str = """
        Check product stock levels in inventory. 
        Use this tool to:
        - Check stock quantity for specific products by name or SKU
        - Find low stock items
        - Check stock status (in_stock, low_stock, out_of_stock)
        - Get inventory details for products

        Input should be a product name, SKU, or search query.
    """
    args_schema: type = StockCheckInput
    data: pd.DataFrame = Field(default_factory=pd.DataFrame)

    def __init__(self, csv_data: pd.DataFrame):
        super().__init__()
        self.data = csv_data

    def _run(self, query: str) -> str:
        try:
            query = query.strip().lower()
            matches = self._search_products(query)

            if matches.empty:
                return f"No products found matching '{query}'. Please check the product name or SKU."
            return self._format_stock_results(matches, query)
        except Exception as e:
            return f"Error checking stock: {str(e)}"

    def _search_products(self, query: str) -> pd.DataFrame:
        df = self.data.copy()
        search_columns = ['name', 'sku', 'brand', 'description', 'short_description']
        mask = pd.Series([False] * len(df))

        for col in search_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower()
                mask |= df[col].str.contains(query, na=False, regex=False)

        return df[mask]

    def _format_stock_results(self, matches: pd.DataFrame, query: str) -> str:
        if len(matches) == 0:
            return f"No products found for '{query}'"

        result = f"Stock Information for '{query}':\n"
        result += "=" * 50 + "\n\n"

        for idx, row in matches.iterrows():
            result += f"Product: {row.get('name', 'N/A')}\n"
            result += f"SKU: {row.get('sku', 'N/A')}\n"
            result += f"Brand: {row.get('brand', 'N/A')}\n"

            quantity = row.get('quantity', 0)
            stock_status = row.get('stock_status', 'unknown')
            low_stock_threshold = row.get('low_stock_threshold', 10)

            result += f"Current Stock: {quantity} units\n"
            result += f"Stock Status: {stock_status}\n"

            if pd.notna(low_stock_threshold):
                result += f"Low Stock Threshold: {low_stock_threshold} units\n"

            if pd.notna(row.get('price')):
                result += f"Price: ${row.get('price', 0):.2f}\n"

            if pd.notna(row.get('supplier_sku')):
                result += f"Supplier SKU: {row.get('supplier_sku')}\n"

            if pd.notna(row.get('lead_time')):
                result += f"Lead Time: {row.get('lead_time')} days\n"

            result += "\n" + "-" * 30 + "\n\n"

        if len(matches) > 1:
            total_stock = matches['quantity'].sum() if 'quantity' in matches.columns else 0
            low_stock_count = len(matches[matches[
                                              'stock_status'] == 'low_stock']) if 'stock_status' in matches.columns else 0
            out_of_stock_count = len(matches[matches[
                                                 'stock_status'] == 'out_of_stock']) if 'stock_status' in matches.columns else 0

            result += f"SUMMARY:\n"
            result += f"Total Products Found: {len(matches)}\n"
            result += f"Total Stock: {total_stock} units\n"
            result += f"Low Stock Items: {low_stock_count}\n"
            result += f"Out of Stock Items: {out_of_stock_count}\n"

        return result