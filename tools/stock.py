import pandas as pd
from typing import List, Dict
from langchain_core.tools import BaseTool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class StockCheckInput(BaseModel):
    query: str = Field(description="Product search query")

class VectorProductStockTool(BaseTool):
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
    vectorstore: Chroma = Field(default=None)
    csv_data: pd.DataFrame = Field(default_factory=pd.DataFrame)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, vectorstore: Chroma, csv_data: pd.DataFrame):
        super().__init__(vectorstore=vectorstore, csv_data=csv_data)

    def _run(self, query: str) -> str:
        try:
            query = query.strip().lower()
            relevant_docs = self._vector_search(query)

            if not relevant_docs:
                return f"No products found matching '{query}'. Please check the product name or SKU."

            products_info = self._extract_product_info(relevant_docs)
            return self._format_stock_results(products_info, query)
        except Exception as e:
            return f"Error checking stock: {str(e)}"

    def _vector_search(self, query: str, k:int=10) -> List[Document]:
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            return [Document(page_content=f"Error during vector search: {str(e)}")]

    def _extract_product_info(self, docs: List[Document]) -> List[Dict]:
        products = []
        seen_skus = set()

        for doc in docs:
            try:
                metadata = doc.metadata
                content = doc.page_content
                product_data = self._parse_product_content(content, metadata)
                if product_data and product_data.get("sku") not in seen_skus:
                    seen_skus.add(product_data.get("sku"))
                    products.append(product_data)
            except Exception as e:
                continue

        return products

    def _parse_product_content(self, content: str, metadata: Dict) -> Dict:
        try:
            lines = content.strip().split('\n')
            product_data = {}

            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()

                    if key in ['quantity', 'low_stock_threshold', 'lead_time', 'category_id']:
                        try:
                            product_data[key] = int(
                                float(value)) if value and value.lower() not in ['nan',
                                                                                 'n/a'] else 0
                        except:
                            product_data[key] = 0
                    elif key in ['price', 'cost', 'compare_at_price', 'weight']:
                        try:
                            product_data[key] = float(value) if value and value.lower() not in [
                                'nan', 'n/a'] else 0.0
                        except:
                            product_data[key] = 0.0
                    else:
                        product_data[key] = value if value.lower() not in ['nan', 'n/a'] else ''

            if metadata:
                product_data.update(metadata)

            return product_data

        except Exception as e:
            return {}


    def _search_products(self, query: str) -> pd.DataFrame:
        df = self.data.copy()
        search_columns = ['name', 'sku', 'brand', 'description', 'short_description']
        mask = pd.Series([False] * len(df))

        for col in search_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower()
                mask |= df[col].str.contains(query, na=False, regex=False)

        return df[mask]

    def _format_stock_results(self, products: List[Dict], query: str) -> str:
        if not products:
            return f"No products found for '{query}'"

        result = f"Stock Information for '{query}':\n"
        result += "=" * 50 + "\n\n"

        for product in products:
            result += f"Product: {product.get('name', 'N/A')}\n"
            result += f"SKU: {product.get('sku', 'N/A')}\n"
            result += f"Brand: {product.get('brand', 'N/A')}\n"

            quantity = product.get('quantity', 0)
            stock_status = product.get('stock_status', 'unknown')
            low_stock_threshold = product.get('low_stock_threshold', 10)

            result += f"Current Stock: {quantity} units\n"
            result += f"Stock Status: {stock_status}\n"

            if low_stock_threshold:
                result += f"Low Stock Threshold: {low_stock_threshold} units\n"

            price = product.get('price', 0)
            if price:
                result += f"Price: ${float(price):.2f}\n"

            supplier_sku = product.get('supplier_sku', '')
            if supplier_sku:
                result += f"Supplier SKU: {supplier_sku}\n"

            lead_time = product.get('lead_time', '')
            if lead_time:
                result += f"Lead Time: {lead_time} days\n"

            short_desc = product.get('short_description', '')
            if short_desc:
                result += f"Description: {short_desc[:100]}...\n"

            result += "\n" + "-" * 30 + "\n\n"

        if len(products) > 1:
            total_stock = sum(p.get('quantity', 0) for p in products)
            low_stock_count = sum(1 for p in products if p.get('stock_status') == 'low_stock')
            out_of_stock_count = sum(1 for p in products if p.get('stock_status') == 'out_of_stock')

            result += f"SUMMARY:\n"
            result += f"Total Products Found: {len(products)}\n"
            result += f"Total Stock: {total_stock} units\n"
            result += f"Low Stock Items: {low_stock_count}\n"
            result += f"Out of Stock Items: {out_of_stock_count}\n"

        return result