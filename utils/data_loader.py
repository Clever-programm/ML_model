import re
from pathlib import Path

def load_texts_from_folder(folder_path, encoding='utf-8', min_length=100):
    """
    Загружает все .txt файлы из папки и объединяет в один текст.
    
    Args:
        folder_path: Путь к папке с файлами
        encoding: Кодировка файлов
        min_length: Минимальная длина текста (фильтр мусора)
    
    Returns:
        str: Объединённый текст
    """
    folder = Path(folder_path)
    texts = []
    
    for file_path in folder.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read().strip()
                if len(content) >= min_length:
                    texts.append(content)
                    print(f"✓ {file_path.name}: {len(content):,} символов")
        except Exception as e:
            print(f"✗ Ошибка чтения {file_path.name}: {e}")
    
    combined = "\n\n---\n\n".join(texts)
    print(f"\nИтого: {len(texts)} файлов, {len(combined):,} символов")
    
    return combined


def analyze_text(text):
    """Простая статистика по тексту"""
    chars = len(text)
    words = len(text.split())
    unique_chars = len(set(text))
    
    print(f"\n📈 Статистика:")
    print(f"   Символов: {chars:,}")
    print(f"   Слов (примерно): {words:,}")
    print(f"   Уникальных символов: {unique_chars}")
    print(f"   Пример текста:\n   '{text[:200]}...'")


def clean_text(text):
    """Базовая очистка текста"""
    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    # Убираем специальные символы (оставляем буквы, цифры, знаки препинания)
    text = re.sub(r'[^\w\s.,!?;:—\-()\"\'«»]', '', text)
    # Убираем пустые строки
    text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
    return text


if __name__ == "__main__":
    text = load_texts_from_folder("learning_data/habr_articles/")
    analyze_text(text)

    with open("data.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("✅ Сохранено в data.txt")