import torch
import torch.nn.functional as F
import json
from model import MiniLLM

from utils.tokenizer import SimpleTokenizer as RealTokenizer

VERSION = "_v2"

# Загрузка модели
def load_model(checkpoint_path, tokenizer_path, device='cuda:0'):
    """
    Загружает модель и токенизатор из сохранённых файлов.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tok_data = json.load(f)
    
    itos_fixed = {int(k): v for k, v in tok_data['itos'].items()}
    tokenizer = RealTokenizer.from_dict(tok_data['stoi'], itos_fixed)
    
    params = checkpoint['model_params']
    model = MiniLLM(
        vocab_size=params['vocab_size'],
        embed_dim=params['embed_dim'],
        n_heads=params['n_heads'],
        n_layers=params['n_layers'],
        seq_length=params['seq_length'],
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, params, tokenizer


# Генерация текста
@torch.no_grad()
def generate_text(model, tokenizer, device, prompt, max_tokens=200, 
                  temperature=0.8, top_k=40, seq_length=64):
    """Генерирует текст, используя модель"""
    model.eval()
    
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

    if tokens.shape[1] == 0:
        tokens = torch.randint(0, tokenizer.vocab_size, (1, 1), device=device)
    
    generated = model.generate(
        tokens, 
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k
    )
    
    return tokenizer.decode(generated[0].tolist())


# Запуск
if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    checkpoint_path = f"checkpoints{VERSION}/checkpoint_best.pt"
    tokenizer_path = f"checkpoints{VERSION}/tokenizer.json"
    
    print("🔄 Загрузка модели...")
    model, params, tokenizer = load_model(checkpoint_path, tokenizer_path, device)
    print(f"✅ Модель загружена: {params['embed_dim']}dim, {params['n_layers']} layers")
    
    print("\n" + "=" * 60)
    print("🤖 MINI-LLM ГЕНЕРАЦИЯ ТЕКСТА")
    print("=" * 60)
    
    # Примеры промптов
    prompts = [
        "Привет, ",
        "Машинное обучение ",
        "Нейронные сети ",
        "Клинические ",
        "",  # Случайная генерация
    ]
    
    for prompt in prompts:
        print(f"\n📝 Промпт: '{prompt}'")
        print("-" * 40)
        output = generate_text(
            model, tokenizer, device, 
            prompt, 
            max_tokens=150,
            temperature=0.8,
            top_k=40,
            seq_length=params['seq_length']
        )
        print(output)