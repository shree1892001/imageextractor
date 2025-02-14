import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

def extract_agent_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    agent_data = []

    if table:
        for tr in table.find_all('tr')[1:]:
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

def main(base_url, search_term, total_pages=431):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    driver.get(base_url)

    search_box = driver.find_element(By.ID, 'SearchBox')
    search_box.send_keys(search_term)
    search_box.send_keys(Keys.RETURN)
    time.sleep(5)

    agent_data = []

    for page in range(total_pages):
        print(f"Processing page {page + 1} of {total_pages}")

        html_content = driver.page_source
        agent_data.extend(extract_agent_data(html_content))

        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, 'pnnext'))
            )
            next_button.click()
            time.sleep(5)
        except Exception as e:
            print(f"Error navigating to the next page: {e}")
            break

    driver.quit()
    return agent_data

if __name__ == "__main__":
    base_url = 'https://web.sos.ky.gov/BusSearchNProfile/RAsearch.aspx'
    search_term = 'registered agent'
    results = main(base_url, search_term)
    save_results_to_csv(results, "scraped_results.csv")