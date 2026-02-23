import os
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests
import json
from requests.compat import urlparse


class Crawler:
    def __init__(
        self, 
        output_dir="pages", 
        index_file="index.txt", 
        max_pages=100, 
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
        saved_json_path="links.json"
    ):
        self.output_dir = output_dir
        self.index_file = index_file
        self.max_pages = max_pages
        self.headers = headers
        self.saved_json_path = saved_json_path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run_crawler_from_list(self, urls: list):
        queue = urls
        count = 1
        with open(self.index_file, "w", encoding="utf-8") as index_file:
            while queue:
                url = queue.pop(0)
                try:
                    response = requests.get(url, headers=self.headers, timeout=10)
                    response.raise_for_status()

                    file_name = f"{count}.txt"
                    file_path = os.path.join(self.output_dir, file_name)
                    
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    index_file.write(f"{count} {url}\n")
                    print(f"[{count}] Успешно скачано: {url}")
                    count += 1
                except Exception as e:                
                    print(f"Ошибка при загрузке {url}: {e}")
                    queue.append(url)

    def run_crawler_with_gen_urls(self, start_url):
        visited = set()
        queue = [start_url]
        count = 1
        
        with open(self.index_file, "w", encoding="utf-8") as index_file:
            while queue and count <= self.max_pages:
                url = queue.pop(0)

                if url in visited:
                    continue

                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }
                    response = requests.get(url, headers=headers, timeout=5)
                    print(response.status_code, url)

                    # Проверяем, что это HTML-страница, а не файл/картинка/js/css
                    if "text/html" not in response.headers.get("Content-Type", ""):
                        continue

                    visited.add(url)

                    html_content = response.text
                    file_name = f"{count}.txt"
                    file_path = os.path.join(self.output_dir, file_name)
                    
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(html_content)

                    index_file.write(f"{count} {url}\n")
                    print(f"[{count}/{self.max_pages}] Успешно скачано: {url}")

                    soup = BeautifulSoup(html_content, "html.parser")
                    for link in soup.find_all("a", href=True):
                        next_url = urljoin(url, link["href"])
                        
                        # Фильтруем ссылки: остаемся только на Википедии и убираем якоря (#)
                        if urlparse(next_url).netloc == urlparse(start_url).netloc:
                            next_url = next_url.split("#")[0]
                            
                            # Добавляем в очередь, если еще не видели
                            if next_url not in visited and next_url not in queue:
                                queue.append(next_url)

                    count += 1

                except Exception as e:
                    print(f"Ошибка при загрузке {url}: {e}")

        with open(self.saved_json_path, "w", encoding="utf-8") as f:
            json.dump(list(visited), f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # crawler = Crawler(
    #     output_dir="gen/pages",
    #     index_file="gen/index.txt",
    #     saved_json_path="gen/links.json"
    # )
    
    # crawler.run_crawler_with_gen_urls("https://en.wikipedia.org/wiki/Web_crawler")
    
    with open("tasks/one/links.json", "r", encoding="utf-8") as file:
        urls = json.load(file)
    
    crawler = Crawler(
        output_dir="pages",
        index_file="index.txt",
    )
    crawler.run_crawler_from_list(urls)
