from langchain.tools import BaseTool
from googleapiclient.discovery import build
import streamlit as st
from typing import Optional, Type
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import lru_cache
import time

class GoogleBooksSchema(BaseModel):
    query: str = Field(..., description="The search query for books about waste management")

class GoogleBooksTool(BaseTool):
    name: str = "google_books"
    description: str = "Search for authoritative books on waste management topics"
    args_schema: Type[BaseModel] = GoogleBooksSchema
    
    # Combined cache and rate limiting
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    @lru_cache(maxsize=100)
    def _run(self, query: str, **kwargs) -> str:
        """Search Google Books API with caching and rate limiting"""
        try:
            # API call
            service = build("books", "v1", developerKey=st.secrets["GOOGLE_BOOKS_API_KEY"])
            results = service.volumes().list(
                q=f"{query} waste management",
                maxResults=3,
                printType="books",
                orderBy="relevance"
            ).execute()
            
            # Process results
            items = results.get("items", [])
            if not items:
                return "No relevant books found"
                
            return "\n".join(
                f"{i+1}. {item['volumeInfo']['title']} by {item['volumeInfo'].get('authors', ['Unknown'])[0]}\n"
                f"   📘 https://books.google.com/books?id={item['id']}\n"
                for i, item in enumerate(items)
            )
            
        except Exception as e:
            time.sleep(2)  # Additional delay on failure
            raise e

    async def _arun(self, query: str, **kwargs) -> str:
        """Async implementation with same protections"""
        return self._run(query, **kwargs)