class SimpleTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        """Текст -> список чисел"""
        return [self.stoi[ch] for ch in text]
    
    def decode(self, ids):
        """Список чисел -> текст"""
        return ''.join([self.itos[id] for id in ids])
    
    @classmethod
    def from_dict(cls, stoi, itos):
        """Создаёт токенизатор из сохранённых словарей"""
        instance = cls.__new__(cls)
        instance.stoi = stoi
        instance.itos = itos
        instance.vocab_size = len(stoi)
        instance.chars = list(itos.values())
        return instance

# Пример использования
if __name__ == "__main__":
    text = "Привет, мир! Это тестовый текст для нашей LLM."
    tokenizer = SimpleTokenizer(text)
    
    print(f"Текст: {text}")
    print(f"Размер словаря: {tokenizer.vocab_size}")
    print(f"Словарь символов: {tokenizer.chars}")
    
    encoded = tokenizer.encode("Привет")
    print(f"\n'Привет' -> {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"{encoded} -> '{decoded}'")