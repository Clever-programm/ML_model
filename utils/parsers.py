import os
import time
import requests
from bs4 import BeautifulSoup

def parse_habr_articles(ids, output_dir, delay=1.5):
    """
    Парсит статьи с Хабра по списку ID.
    
    :param ids: Список или диапазон ID статей (int)
    :param output_dir: Директория для сохранения txt файлов
    :param delay: Задержка между запросами в секундах (защита от бана)
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Директория {output_dir} создана.")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for article_id in ids:
        url = f"https://habr.com/ru/articles/{article_id}/"
        filename = os.path.join(output_dir, f"{article_id}.txt")
        
        print(f"Парсинг статьи {article_id}...")
        
        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 404:
                print(f"[WARNING] Статья с ID {article_id} не найдена (404). Пропускаем.")
                time.sleep(delay)
                continue
            
            if response.status_code != 200:
                print(f"[WARNING] Ошибка доступа к статье {article_id} (Status: {response.status_code}). Пропускаем.")
                time.sleep(delay)
                continue

            soup = BeautifulSoup(response.text, 'lxml')

            title_tag = soup.find('h1', class_='tm-title tm-title_h1')
            if not title_tag:
                print(f"[WARNING] Заголовок не найден для статьи {article_id}, но продолжаем парсинг тела.")

            body_div = soup.find('div', class_='article-body')
            
            if not body_div:
                print(f"[WARNING] Блок article-body не найден для статьи {article_id}. Пропускаем.")
                time.sleep(delay)
                continue

            for table in body_div.find_all('table'):
                table.decompose()
            
            for tag in body_div(['script', 'style']):
                tag.decompose()

            text = body_div.get_text(separator=' ', strip=True)

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"Успешно сохранено: {filename}")

        except Exception as e:
            print(f"[ERROR] Критическая ошибка при парсинге {article_id}: {e}")
        
        time.sleep(delay)

if __name__ == "__main__":
    target_ids = range(700100, 700200, 2) 
    
    save_directory = "learning_data/habr_articles"
    
    parse_habr_articles(target_ids, save_directory)