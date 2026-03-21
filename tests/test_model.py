import torch
from model import MiniLLM
from train import Config

config = Config()
vocab_size = 178

model = MiniLLM(
    vocab_size, 
    config.embed_dim, 
    config.n_heads, 
    config.n_layers, 
    config.seq_length, 
    config.dropout
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Всего параметров: {total_params:,}")
print(f"Размер модели: {total_params * 4 / 1024**2:.2f} MB")

x = torch.randint(0, vocab_size, (config.batch_size, config.seq_length)).to(device)
print(f"\nВход: {x.shape}")

with torch.no_grad():
    logits = model(x)
print(f"Выход (logits): {logits.shape}")

assert logits.shape == (config.batch_size, config.seq_length, vocab_size), "Неправильная форма выхода!"

print("\nМодель работает корректно!")

print("\nТест генерации текста:")
start_token = torch.tensor([[vocab_size // 2]]).to(device)
generated = model.generate(start_token, max_new_tokens=50, temperature=0.8)
print(f"Сгенерировано токенов: {generated.shape[1]}")