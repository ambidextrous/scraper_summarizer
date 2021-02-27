from typing import List, Dict, Tuple, Any

from bs4 import BeautifulSoup
from transformers import pipeline
from requests_html import HTMLSession
import pandas as pd

INPUT_CSV = "urls.csv"
OUTPUT_CSV = "url_texts.csv"

ELEMENT_EXTRACTION_FUNCTION = lambda x: x.find_all("p", {"class": "featured__copy"})


def scrape(url, element_extraction_function):
    session = HTMLSession()
    headers = {
        "content-language": "en",
        "Accept-Language": "en-US",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
    }
    r = session.get(url, headers=headers, allow_redirects=True)
    print(f"request={r}")
    print(f"history={r.history}")
    t = r.text
    print(f"head={t[:280]}")
    print(f"tail={t[-280:]}")
    soup = BeautifulSoup(t)
    text = ". ".join([item.getText() for item in element_extraction_function(soup)]).replace("..", ".")
    return text


def read_input_csv(csv_filename: str) -> pd.DataFrame:
    df = pd.read_csv(csv_filename)
    return df


def get_urls_from_csv(csv_filename: str) -> List[str]:
    df = read_input_csv(csv_filename)
    text_list = df["url"].tolist()
    return text_list


def process_scraping(input_csv: str, output_csv: str):
    url_list = get_urls_from_csv(input_csv)
    print(f"url_list={url_list}")
    processed_data = [
        {
            "url": url,
            "text": scrape(url, ELEMENT_EXTRACTION_FUNCTION),
        }
        for url in url_list
    ]
    df = pd.DataFrame(processed_data)
    print(df)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    process_scraping(INPUT_CSV, OUTPUT_CSV)
