import asyncio
import pandas as pd
import aiohttp
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Any, Annotated
from langgraph.graph import Graph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from Common.constants import *


class SearchResult(BaseModel):
    """Schema for search results"""
    url: str = Field(description="The URL of the search result")
    title: str = Field(description="The title of the search result")
    description: str = Field(description="Brief description of the search result")


class WebScraper:
    """Class responsible for web scraping operations"""

    def __init__(self):
        self.parser = ContentParser()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_content(self, url: str) -> str:
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        async with self.session.get(url) as response:
            return await response.text()

    async def extract_info(self, url: str) -> Dict:
        try:
            content = await self.fetch_content(url)
            soup = BeautifulSoup(content, 'html.parser')
            return self.parser.parse(soup)
        except Exception as e:
            print(f"Error extracting info from {url}: {e}")
            return {}


class ContentParser:
    """Class for parsing website content"""

    def parse(self, soup: BeautifulSoup) -> Dict:
        return {
            'email': self._parse_emails(soup),
            'phone': self._parse_phones(soup),
            'social_links': self._parse_social_links(soup)
        }

    def _parse_emails(self, soup: BeautifulSoup) -> List[str]:
        return [a['href'][7:] for a in soup.find_all('a', href=True)
                if 'mailto:' in a['href']]

    def _parse_phones(self, soup: BeautifulSoup) -> List[str]:
        return [a.text for a in soup.find_all('a', href=True)
                if 'tel:' in a['href']]

    def _parse_social_links(self, soup: BeautifulSoup) -> Dict[str, str]:
        social_platforms = {
            'facebook.com': 'facebook',
            'twitter.com': 'twitter',
            'linkedin.com': 'linkedin',
            'instagram.com': 'instagram'
        }

        social_links = {}
        for a in soup.find_all('a', href=True):
            href = a['href']
            for platform_url, platform_name in social_platforms.items():
                if platform_url in href:
                    social_links[platform_name] = href
        return social_links


class SearchAutomator:
    def __init__(self, api_key: str):
        self.api_client = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=1000,
            max_retries=3,
            google_api_key=api_key
        )
        self.scraper = WebScraper()

    def create_search_workflow(self) -> Graph:
        # Define the nodes in our workflow
        def search_web(state: Dict) -> Dict:
            """Node that performs web search using LLM"""
            query = state["query"]
            page = state.get("page", 1)

            messages = [
                SystemMessage(content="""You are a search assistant. Return exactly 10 search results in JSON format.
                Each result must have url, title, and description fields. 
                Format your response as a valid JSON object with a 'results' array."""),
                HumanMessage(content=f"Search for: {query} (page {page})")
            ]

            response = self.api_client.invoke(messages)
            try:
                # Parse the JSON response
                json_response = json.loads(response.content)
                # Convert each result to a SearchResult model
                search_results = [SearchResult(**result) for result in json_response["results"]]
                return {"search_results": search_results, "page": page}
            except Exception as e:
                print(f"Error parsing search results: {e}")
                print(f"Raw response: {response.content}")
                return {"search_results": [], "page": page}

        async def extract_contact_info(state: Dict) -> Dict:
            """Node that extracts contact information from search results"""
            results = state["search_results"]
            async with self.scraper as scraper:
                contact_info = []
                for result in results:
                    info = await scraper.extract_info(result.url)
                    contact_info.append({
                        "url": result.url,
                        "title": result.title,
                        "contact_info": info
                    })
            return {"contact_info": contact_info}

        # Create the workflow graph
        workflow = Graph()

        # Add nodes
        workflow.add_node("search", search_web)
        workflow.add_node("extract", extract_contact_info)

        # Define the entry point and edges
        workflow.set_entry_point("search")
        workflow.add_edge("search", "extract")
        workflow.add_edge("extract", END)

        return workflow.compile()

    async def run_search(self, query: str, num_pages: int = 12) -> List[Dict]:
        workflow = self.create_search_workflow()
        all_results = []

        for page in range(1, num_pages + 1):
            print(f"Processing page {page}...")
            try:
                results = await workflow.ainvoke({
                    "query": query,
                    "page": page
                })
                if "contact_info" in results:
                    all_results.extend(results["contact_info"])
            except Exception as e:
                print(f"Error processing page {page}: {e}")
            await asyncio.sleep(2)

        return all_results

    def save_results(self, results: List[Dict], filename: str):
        if not results:
            return

        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)


async def main():
    api_key = API_KEY# Replace with your actual API key
    query = AUTOMATION_TASK  # Replace with your search query

    automator = SearchAutomator(api_key)
    results = await automator.run_search(query)
    automator.save_results(results, 'search_results.csv')


if __name__ == "__main__":
    asyncio.run(main())