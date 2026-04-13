"""
Шаг 1: Экспорт BERT в ONNX

Что здесь происходит:
- Загружаем bert-base-uncased из HuggingFace
- Создаём dummy-input — трассировщику нужен реальный тензор
  чтобы пройти по графу и записать все операции
- Экспортируем с dynamic_axes — это позволяет менять
  batch_size и sequence_length во время инференса
"""

import torch
from transformers import BertModel, BertTokenizer

# ─── Конфиг ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "bert-base-uncased"
ONNX_PATH   = "bert.onnx"
MAX_SEQ_LEN = 128
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

print(f"[1/4] Загружаем {MODEL_NAME}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model     = BertModel.from_pretrained(MODEL_NAME)
model.eval()
model.to(DEVICE)

# Dummy-input: любой реальный текст, главное — форма тензора
# torch.onnx.export прогоняет этот пример через модель и записывает граф
sample_text = "The quick brown fox jumps over the lazy dog"
encoding    = tokenizer(
    sample_text,
    max_length=MAX_SEQ_LEN,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

input_ids      = encoding["input_ids"].to(DEVICE)       # [1, 128]
attention_mask = encoding["attention_mask"].to(DEVICE)  # [1, 128]
token_type_ids = encoding["token_type_ids"].to(DEVICE)  # [1, 128]

print(f"[2/4] Трассируем граф (input shape: {input_ids.shape})...")

with torch.no_grad():
    torch.onnx.export(
        model,
        # Все входы модели передаём как кортеж
        args=(input_ids, attention_mask, token_type_ids),

        f=ONNX_PATH,

        # Имена входов/выходов — важны для onnxruntime
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state", "pooler_output"],

        # dynamic_axes говорит экспортёру: эти размерности могут меняться.
        # Без этого модель будет зафиксирована на batch=1, seq=128.
        dynamic_axes={
            "input_ids":      {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
            "pooler_output":     {0: "batch_size"},
        },

        # opset 14 — хороший баланс: поддерживается везде,
        # включает оптимизации для attention-операций
        opset_version=14,

        # Убирает ненужные отладочные данные из .onnx файла
        do_constant_folding=True,

        # Подробный лог — полезно при первом экспорте
        verbose=False,
    )

print(f"[3/4] Файл сохранён: {ONNX_PATH}")

# ─── Быстрая проверка структуры графа ─────────────────────────────────────────
import onnx
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)  # бросит исключение если граф сломан

print(f"[4/4] Граф валиден")
print(f"      Входы:  {[i.name for i in onnx_model.graph.input]}")
print(f"      Выходы: {[o.name for o in onnx_model.graph.output]}")
print(f"      Узлов в графе: {len(onnx_model.graph.node)}")

# Считаем размер файла
import os
size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
print(f"      Размер файла: {size_mb:.1f} MB")
print()
print("Готово! Теперь запускай step2_validate.py")
