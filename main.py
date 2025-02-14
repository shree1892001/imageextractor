# import asyncio
# from langchain_google_genai import ChatGoogleGenerativeAI
# from browser_use import Agent
# import pandas as pd
# from Common.constants import API_KEY
# from bs4 import BeautifulSoup
# import aiohttp
# import re
# from urllib.parse import urljoin
#
#
# async def fetch_content(session, url):
#     async with session.get(url) as response:
#         return await response.text()
#
#
# def parse_emails(soup):
#     return [a['href'][7:] for a in soup.find_all('a', href=True) if 'mailto:' in a['href']]
#
#
# def parse_phone_numbers(soup):
#     return [a.text for a in soup.find_all('a', href=True) if 'tel:' in a['href']]
#
#
# def parse_social_links(soup):
#     social_links = {}
#     for a in soup.find_all('a', href=True):
#         href = a['href']
#         if 'facebook.com' in href:
#             social_links['facebook'] = href
#         elif 'twitter.com' in href:
#             social_links['twitter'] = href
#         elif 'linkedin.com' in href:
#             social_links['linkedin'] = href
#         elif 'instagram.com' in href:
#             social_links['instagram'] = href
#     return social_links
#
#
# def parse_contact_info(soup):
#     contact_info = {}
#     contact_info['email'] = parse_emails(soup)
#     contact_info['phone'] = parse_phone_numbers(soup)
#     contact_info['social_links'] = parse_social_links(soup)
#     return contact_info
#
#
# async def extract_contact_info(url, session):
#     try:
#         content = await fetch_content(session, url)
#         soup = BeautifulSoup(content, 'html.parser')
#         contact_info = parse_contact_info(soup)
#         return contact_info
#     except Exception as e:
#         print(f"Error extracting contact info from {url}: {e}")
#         return {}
#
#
# async def fetch_all_pages(session, base_url):
#     all_content = []
#     current_url = base_url
#
#     while current_url:
#         content = await fetch_content(session, current_url)
#         soup = BeautifulSoup(content, 'html.parser')
#         all_content.append(soup)
#
#         next_button = soup.find('a', id='pnnext')
#         if next_button and 'href' in next_button.attrs:
#             current_url = urljoin(base_url, next_button['href'])
#         else:
#             current_url = None
#
#     return all_content
#
#
# async def perform_search_and_scrape(query, api_key):
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",
#         temperature=0,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#         google_api_key=api_key
#     )
#
#     agent = Agent(
#         task=query,
#         llm=llm,
#     )
#
#     result = await agent.run()
#     extracted_content = result.extracted_content()
#
#     async with aiohttp.ClientSession() as session:
#         for item in extracted_content:
#             url = item.get('URL')
#             if url:
#                 pages_content = await fetch_all_pages(session, url)
#                 all_contact_info = []
#
#                 for soup in pages_content:
#                     contact_info = parse_contact_info(soup)
#                     all_contact_info.append(contact_info)
#
#                 combined_contact_info = {
#                     'email': sum([info['email'] for info in all_contact_info], []),
#                     'phone': sum([info['phone'] for info in all_contact_info], []),
#                     'social_links': {k: v for info in all_contact_info for k, v in info['social_links'].items()}
#                 }
#
#                 # Flatten the contact information
#                 item['Emails'] = '; '.join(combined_contact_info['email']) if combined_contact_info['email'] else ''
#                 item['Phone_Numbers'] = '; '.join(combined_contact_info['phone']) if combined_contact_info[
#                     'phone'] else ''
#                 item['Facebook'] = combined_contact_info['social_links'].get('facebook', '')
#                 item['Twitter'] = combined_contact_info['social_links'].get('twitter', '')
#                 item['LinkedIn'] = combined_contact_info['social_links'].get('linkedin', '')
#                 item['Instagram'] = combined_contact_info['social_links'].get('instagram', '')
#
#                 # Remove the nested contact info dictionary
#                 if 'Contact Info' in item:
#                     del item['Contact Info']
#
#     return extracted_content
#
#
# def save_results_to_csv(results, filename="scraped_results.csv"):
#     if not results:
#         print("No results to save.")
#         return
#
#     try:
#         # Convert results to DataFrame
#         df = pd.DataFrame(results)
#
#         # Ensure all required columns exist
#         required_columns = ['URL', 'Name', 'Emails', 'Phone_Numbers',
#                             'Facebook', 'Twitter', 'LinkedIn', 'Instagram']
#
#         for col in required_columns:
#             if col not in df.columns:
#                 df[col] = ''
#
#         # Reorder columns
#         df = df[required_columns]
#
#         # Clean the data
#         df = df.fillna('')
#
#         # Save to CSV
#         df.to_csv(filename, index=False, encoding='utf-8')
#         print(f"\nScraping complete. {len(results)} records saved to '{filename}'.")
#
#         # Print summary
#         print("\nData Summary:")
#         print(f"Total records: {len(df)}")
#         print(f"Records with emails: {df['Emails'].str.len().gt(0).sum()}")
#         print(f"Records with phone numbers: {df['Phone_Numbers'].str.len().gt(0).sum()}")
#         print(
#             f"Records with social media links: {df[['Facebook', 'Twitter', 'LinkedIn', 'Instagram']].notna().any(axis=1).sum()}")
#
#     except Exception as e:
#         print(f"Error saving results to CSV: {e}")
#
#
# if __name__ == "__main__":
#     while (True):
#         search_query = input("Enter your search query (or 'q' to quit): ")
#         if search_query.lower() == 'q':
#             break
#
#         api_key = API_KEY
#
#         try:
#             loop = asyncio.get_event_loop()
#             results = loop.run_until_complete(perform_search_and_scrape(search_query, api_key))
#
#             save_results_to_csv(results, "scraped_results.csv")
#
#             # Print detailed results
#             print("\nDetailed Results:")
#             for result in results:
#                 print("\nEntry:")
#                 print(f"URL: {result['URL']}")
#                 print(f"Name: {result.get('Name', 'N/A')}")
#                 print(f"Emails: {result.get('Emails', 'N/A')}")
#                 print(f"Phone Numbers: {result.get('Phone_Numbers', 'N/A')}")
#                 print("Social Media Links:")
#                 print(f"  Facebook: {result.get('Facebook', 'N/A')}")
#                 print(f"  Twitter: {result.get('Twitter', 'N/A')}")
#                 print(f"  LinkedIn: {result.get('LinkedIn', 'N/A')}")
#                 print(f"  Instagram: {result.get('Instagram', 'N/A')}")
#                 print("-" * 50)
#
#         except Exception as e:
#             print(f"Error during scraping: {e}")


import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
import pandas as pd
from Common.constants import *
from bs4 import BeautifulSoup
import aiohttp

async def fetch_content(session, url):
    async with session.get(url) as response:
        return await response.text()

async def extract_contact_info(url, session):
    try:
        content = await fetch_content(session, url)
        soup = BeautifulSoup(content, 'html.parser')

        contact_info = {}
        contact_info['email'] = [a['href'][7:] for a in soup.find_all('a', href=True) if 'mailto:' in a['href']]
        contact_info['phone'] = [a.text for a in soup.find_all('a', href=True) if 'tel:' in a['href']]

        return contact_info
    except Exception as e:
        print(f"Error extracting contact info from {url}: {e}")
        return {}

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

    async with aiohttp.ClientSession() as session:
        for item in extracted_content:
            url = item.get('URL')
            if url:
                contact_info = await extract_contact_info(url, session)
                item['Contact Info'] = contact_info

    return extracted_content

def save_results_to_csv(results, filename="website_data.csv"):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Scraping complete. {len(results)} records saved to '{filename}'.")

if __name__ == "__main__":
    search_query = f'''
    
    
    '''

    api_key = API_KEY

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(perform_search_and_scrape(search_query, api_key))

    save_results_to_csv(results, "website_data.csv")

    for result in results:
        print(f"URL: {result['URL']}\nName: {result['Name']}\nContact Info: {result['Contact Info']}\n")