import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
import pandas as pd
from Common.constants import API_KEY
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

def extract_agent_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    agent_data = []

    if table:
        for tr in table.find_all('tr')[1:]:  # Skip header row
            cells = tr.find_all('td')
            if cells:
                agent_name = cells[0].get_text(strip=True)
                llc_name = cells[1].get_text(strip=True)
                address = cells[2].get_text(strip=True)
                agent_data.append({'Registered Agent': agent_name, 'LLC Name': llc_name, 'Address': address})

    return agent_data

def save_results_to_csv(results, filename="scraped_results.csv"):
    if not results:
        print("No results to save.")
        return

    try:
        df = pd.DataFrame(results)
        df.drop_duplicates(subset=['Registered Agent'], inplace=True)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\nScraping complete. {len(df)} unique records saved to '{filename}'.")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

async def fetch_all_pages(driver, total_pages=431):
    agent_data = []

    for page in range(total_pages):
        print(f"Processing page {page + 1} of {total_pages}")

        html_content = driver.page_source
        agent_data.extend(extract_agent_data(html_content))

        try:
            # Locate the "Next" button by its class
            next_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'span.oeN89d'))
            )
            next_button.click()
            time.sleep(5)
        except Exception as e:
            print(f"Error navigating to the next page: {e}")
            break

    return agent_data

async def perform_search_and_scrape(query, api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=api_key
    )

    agent = Agent(
        task=query,
        llm=llm,
    )

    result = await agent.run()
    extracted_content = result.extracted_content()

    # Retrieve the URLs from the search results
    urls = extract_urls_from_search_results(extracted_content)

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

    all_agent_data = []

    for url in urls:
        driver.get(url)
        time.sleep(5)  # Ensure page is fully loaded

        agent_data = await fetch_all_pages(driver)
        all_agent_data.extend(agent_data)

    driver.quit()

    return all_agent_data

def extract_urls_from_search_results(search_results):
    soup = BeautifulSoup(search_results, 'html.parser')
    links = soup.find_all('a', href=True)

    urls = []
    for link in links:
        url = link['href']
        if url.startswith("http"):
            urls.append(url)

    return urls

if __name__ == "__main__":
    while True:
        search_query = "west virginia registered agents"
        if search_query.lower() == 'q':
            break

        api_key = API_KEY

        try:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(perform_search_and_scrape(search_query, api_key))

            save_results_to_csv(results, "scraped_results.csv")

        except Exception as e:
            print(f"Error during scraping: {e}")