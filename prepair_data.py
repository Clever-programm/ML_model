import torch

from tokenizer import SimpleTokenizer

text = "\n".join(open("./text.txt", "r", encoding="utf-8").readlines())

tokenizer = SimpleTokenizer(text)
print(f"Размер словаря: {tokenizer.vocab_size}")

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
print(f"Всего токенов: {len(data)}")

def create_batches(data, seq_length, batch_size=4):
    """Создаём батчи для обучения LLM"""
    n_batches = (len(data) - seq_length - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        x = data[start_idx:start_idx + batch_size * seq_length].reshape(batch_size, seq_length)
        y = data[start_idx + 1:start_idx + 1 + batch_size * seq_length].reshape(batch_size, seq_length)
        yield x, y

print("\n" + "="*50)
for seq_len in [16, 32, 64]:
    x_batch, y_batch = next(create_batches(data, seq_length=seq_len))
    print(f"seq_length={seq_len}: X.shape={x_batch.shape}, Y.shape={y_batch.shape}")
    
print("="*50)

print("\nПример батча (seq_length=32):")
x_batch, y_batch = next(create_batches(data, seq_length=32))
print(f"Вход (X): {x_batch[0, :20].tolist()}")
print(f"Цель (Y): {y_batch[0, :20].tolist()}")

print(f"\nТекст входа:  '{tokenizer.decode(x_batch[0, :20].tolist())}'")
print(f"Текст цели:   '{tokenizer.decode(y_batch[0, :20].tolist())}'")