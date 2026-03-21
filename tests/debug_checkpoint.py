import torch

checkpoint_path = "checkpoints_v2/checkpoint_best.pt"

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"✅ Файл загружен")
    print(f"📊 Тип объекта: {type(checkpoint)}")
    print(f"📊 Содержимое: {checkpoint}")
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")

# Также проверь, какие файлы есть в папке
import os
print("\n📁 Файлы в checkpoints_v2/:")
for f in os.listdir("checkpoints_v2/"):
    path = os.path.join("checkpoints_v2/", f)
    size = os.path.getsize(path)
    print(f"   {f}: {size:,} байт")