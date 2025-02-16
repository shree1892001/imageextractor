import asyncio
import pandas as pd
import aiohttp
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from bs4 import BeautifulSoup
from Common.constants import *
from typing import Dict, List, Optional


class APIClientSingleton:
    _instance = None

    def __new__(cls, api_key: str):
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
        self.api_client = APIClientSingleton(api_key)
        self.parser = ContentParser()

    def create_search_agent(self) -> Agent:
        return Agent(
            role='Web Researcher',
            goal='Find and analyze relevant web pages based on search queries',
            backstory='Expert at web search and information gathering',
            llm=self.api_client.llm,
            tools=['web-search']  # CrewAI's built-in web search tool
        )

    def create_scraping_agent(self) -> Agent:
        return Agent(
            role='Web Scraper',
            goal='Extract contact information from web pages',
            backstory='Specialist in parsing and extracting structured data from websites',
            llm=self.api_client.llm,
            tools=['web-browser']  # CrewAI's built-in web browsing tool
        )

    async def process_urls(self, urls: List[str], page_num: int) -> List[Dict]:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for url in urls:
                tasks.append(self.extract_info(session, url, page_num))
            return await asyncio.gather(*tasks)

    async def extract_info(self, session: aiohttp.ClientSession, url: str, page_num: int) -> Dict:
        try:
            async with session.get(url) as response:
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                contact_info = self.parser.parse(soup)
                return {
                    'URL': url,
                    'Contact Info': contact_info,
                    'Page': page_num
                }
        except Exception as e:
            print(f"Error extracting info from {url}: {e}")
            return {}

    def run_search(self, query: str, num_pages: int = 12) -> List[Dict]:
        # Create agents
        search_agent = self.create_search_agent()
        scraping_agent = self.create_scraping_agent()

        all_results = []
        for page_num in range(1, num_pages + 1):
            print(f"Processing page {page_num}...")

            # Create tasks for the crew
            search_task = Task(
                description=f"Search for: {query} page {page_num}",
                agent=search_agent
            )

            scrape_task = Task(
                description="Extract contact information from the found URLs",
                agent=scraping_agent,
                depends_on=[search_task]
            )

            # Create and run the crew
            crew = Crew(
                agents=[search_agent, scraping_agent],
                tasks=[search_task, scrape_task],
                process=Process.sequential  # Tasks run in sequence
            )

            result = crew.kickoff()

            # Process results
            if isinstance(result, list):
                results = asyncio.run(self.process_urls(result, page_num))
                all_results.extend(results)

            time.sleep(2)  # Respect rate limits

        return all_results

    def save_results(self, results: List[Dict], filename: str):
        if not results:
            return

        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['URL', 'Contact Info', 'Page'])
            writer.writeheader()
            writer.writerows(results)


def main():
    automator = SearchAutomator(API_KEY)
    results = automator.run_search(AUTOMATION_TASK)
    automator.save_results(results, 'search_results.csv')

    for result in results:
        print(f"URL: {result['URL']}\nContact Info: {result['Contact Info']}\n")


if __name__ == "__main__":
    main()