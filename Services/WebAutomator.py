import asyncio
import pandas as pd
import aiohttp
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from bs4 import BeautifulSoup
import re
import time
import csv
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from Common.constants import *

class APIClientSingleton:
    _instance = None

    def __new__(cls,    api_key: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.api_key = api_key
            cls._instance.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0,
                max_tokens=None,
                timeout=1000,
                max_retries=3,
                google_api_key=api_key
            )
        return cls._instance

class ContentParser(ABC):
    """Abstract base class for content parsing strategies"""
    @abstractmethod
    def parse(self, soup: BeautifulSoup) -> Dict:
        pass

class ContactInfoParser(ContentParser):
    """Strategy for parsing contact information"""
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

class WebScraper:
    """Class responsible for web scraping operations"""
    def __init__(self, parser: ContentParser):
        self.parser = parser
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

class SearchAutomator:
    def __init__(self, api_key: str):
        self.api_client = APIClientSingleton(api_key)
        self.parser = ContactInfoParser()

    async def process_page(self, query: str, page_num: int) -> List[Dict]:
        agent = Agent(
            task=query,
            llm=self.api_client.llm,
        )

        result = await agent.run()
        extracted_content = result.extracted_content()

        all_results = []
        async with WebScraper(self.parser) as scraper:
            for item in extracted_content:
                if isinstance(item, dict):
                    url = item.get('URL')
                    if url:
                        contact_info = await scraper.extract_info(url)
                        all_results.append({
                            'URL': url,
                            'Contact Info': contact_info,
                            'Page': page_num
                        })

        return all_results

    async def run_search(self, query: str, num_pages: int = 12) -> List[Dict]:
        all_results = []
        for page_num in range(1, num_pages + 1):
            print(f"Scraping page {page_num}...")
            search_query_page = f"{query} page {page_num}"
            results = await self.process_page(search_query_page, page_num)
            all_results.extend(results)
            await asyncio.sleep(2)
        return all_results

    def save_results(self, results: List[Dict], filename: str):
        if not results:
            return

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['URL', 'Contact Info', 'Page'])
            writer.writeheader()
            writer.writerows(results)

async def main():
    automator = SearchAutomator(API_KEY)
    results = await automator.run_search(AUTOMATION_TASK)
    automator.save_results(results, 'search_results.csv')

    for result in results:
        print(f"URL: {result['URL']}\nContact Info: {result['Contact Info']}\n")

if __name__ == "__main__":
    asyncio.run(main())