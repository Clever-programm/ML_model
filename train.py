import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import os
import json
from tokenizer import SimpleTokenizer
from model import MiniLLM

# Конфигурация обучения
class Config:
    # Данные
    text_file = "text.txt"
    seq_length = 64
    batch_size = 16
    
    # Модель
    vocab_size = None
    embed_dim = 256
    n_heads = 4
    n_layers = 6
    dropout = 0.2
    
    # Обучение
    learning_rate = 2e-4
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Сохранение
    save_dir = "checkpoints"
    save_every = 500

config = Config()

# Dataset для PyTorch
class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return x, y

# Функция обучения на один шаг
def train_step(model, batch_x, batch_y, criterion, optimizer, device):
    model.train()
    
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    logits = model(batch_x)
    
    B, T, V = logits.shape
    loss = criterion(logits.reshape(B * T, V), batch_y.reshape(B * T))
    
    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    
    return loss.item()

# Функция валидации
@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        logits = model(batch_x)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), batch_y.reshape(B * T))
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches

# Генерация текста во время обучения (для мониторинга)
@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt=""):
    model.eval()
    
    if prompt:
        tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    else:
        tokens = torch.randint(0, config.vocab_size, (1, 1), device=device)
    
    generated = model.generate(tokens, max_new_tokens=100, temperature=0.8, top_k=40)
    return tokenizer.decode(generated[0].tolist())

# Главный цикл обучения
def train():
    print("=" * 60)
    print("ОБУЧЕНИЕ MINI-LLM")
    print("=" * 60)
    
    patience = 3
    no_improve_count = 0
    best_val_loss = float('inf')

    # 1. Загрузка и подготовка данных
    print("\nЗагрузка данных...")
    with open(config.text_file, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"   Размер текста: {len(text):,} символов")
    
    # 2. Токенизация
    print("\nТокенизация...")
    tokenizer = SimpleTokenizer(text)
    config.vocab_size = tokenizer.vocab_size
    print(f"   Размер словаря: {config.vocab_size}")
    
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"   Всего токенов: {len(data):,}")
    
    # 3. Разделение на train/val
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"   Train: {len(train_data):,} токенов")
    print(f"   Val:   {len(val_data):,} токенов")
    
    # 4. Создание DataLoader
    train_dataset = TextDataset(train_data, config.seq_length)
    val_dataset = TextDataset(val_data, config.seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"\n   Батчей в эпохе: {len(train_loader)}")
    
    # 5. Создание модели
    print("\nСоздание модели...")
    model = MiniLLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        seq_length=config.seq_length,
        dropout=config.dropout
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Параметры: {total_params:,} ({total_params * 4 / 1024**2:.2f} MB)")
    
    # 6. Оптимизатор и Loss
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * config.epochs
    )
    
    # 7. Создание директории для чекпоинтов
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 8. Цикл обучения
    print("\n" + "=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        total_train_loss = 0
        n_batches = 0
        stop_training = False
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            loss = train_step(model, batch_x, batch_y, criterion, optimizer, config.device)
            total_train_loss += loss
            n_batches += 1
            global_step += 1
            
            scheduler.step()
            
            if batch_idx % 50 == 0:
                avg_loss = total_train_loss / n_batches
                print(f"  Эпоха {epoch+1}/{config.epochs} | "
                      f"Шаг {global_step} | "
                      f"Train Loss: {avg_loss:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if global_step % config.save_every == 0:
                val_loss = validate(model, val_loader, criterion, config.device)
                print(f"\n  Сохранение чекпоинта (step {global_step})...")
                print(f"  Validation Loss: {val_loss:.4f}")
                
                checkpoint = {
                    'config': {
                        'vocab_size': config.vocab_size,
                        'embed_dim': config.embed_dim,
                        'n_heads': config.n_heads,
                        'n_layers': config.n_layers,
                        'seq_length': config.seq_length,
                        'dropout': config.dropout,
                        'text_file': config.text_file,
                    },
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'tokenizer': tokenizer,
                    'stoi': tokenizer.stoi,
                    'itos': tokenizer.itos
                }
                torch.save(checkpoint, f"{config.save_dir}/checkpoint_step_{global_step}.pt")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve_count = 0
                    torch.save(checkpoint, f"{config.save_dir}/checkpoint_best.pt")
                    print(f"  ⭐ Новый лучший результат!")
                else:
                    no_improve_count += 1
                    print(f"  ⏳ Нет улучшений ({no_improve_count}/{patience})")
                    
                if no_improve_count >= patience:
                    print(f"\n🛑 Ранняя остановка! Нет улучшений {patience} эпох подряд.")
                    stop_training = True
                    break
                
                print("\n  Пример генерации:")
                sample = generate_sample(model, tokenizer, config.device, prompt="")
                print(f"  {sample[:200]}...")
                print()
        
        epoch_time = time.time() - epoch_start
        avg_train_loss = total_train_loss / n_batches
        val_loss = validate(model, val_loader, criterion, config.device)
        
        print(f"\n{'='*60}")
        print(f"Эпоха {epoch+1} завершена за {epoch_time:.1f} сек")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"{'='*60}\n")
        
        if stop_training:
            break
    
    # Финальное сохранение
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'model_params': {
            'vocab_size': config.vocab_size,
            'embed_dim': config.embed_dim,
            'n_heads': config.n_heads,
            'n_layers': config.n_layers,
            'seq_length': config.seq_length,
        }
    }, f"{config.save_dir}/checkpoint_best.pt")

    with open(f"{config.save_dir}/tokenizer.json", "w", encoding="utf-8") as f:
        json.dump({
            'stoi': tokenizer.stoi,
            'itos': {str(k): v for k, v in tokenizer.itos.items()}
        }, f, ensure_ascii=False, indent=2)
    
    print("\nОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"Модель сохранена в: {config.save_dir}/")
    
    return model, tokenizer

# Запуск
if __name__ == "__main__":
    train()