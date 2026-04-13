"""
Шаг 2: Валидация — убеждаемся, что ONNX даёт те же числа, что PyTorch

Это ОБЯЗАТЕЛЬНЫЙ шаг перед бенчмарком.
Нельзя сравнивать скорость, не убедившись что результаты идентичны.

Типичные расхождения:
- atol=1e-4 норма для FP32 (погрешности порядка операций)
- atol=1e-2 норма для FP16 (меньше точность)
- Если больше — что-то пошло не так при экспорте
"""

import numpy as np
import torch
import onnxruntime as ort
from transformers import BertModel, BertTokenizer

# ─── Конфиг ───────────────────────────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
ONNX_PATH  = "bert.onnx"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

print("Загружаем модели...")
tokenizer  = BertTokenizer.from_pretrained(MODEL_NAME)
pt_model   = BertModel.from_pretrained(MODEL_NAME).eval().to(DEVICE)

# ─── Создаём ONNX Runtime сессию ─────────────────────────────────────────────
# SessionOptions позволяет тонко настраивать рантайм
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# ORT_ENABLE_ALL включает:
#   - базовые оптимизации (constant folding, dead code elimination)
#   - расширенные (operator fusion, layout optimization)
#   - специфичные для провайдера (cuDNN kernels для CUDA)

# Провайдер: пробуем GPU, падаем на CPU если нет CUDA
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if DEVICE == "cuda" else ["CPUExecutionProvider"]

ort_session = ort.InferenceSession(ONNX_PATH, so, providers=providers)

active_provider = ort_session.get_providers()[0]
print(f"ORT провайдер: {active_provider}")

# ─── Тестовый текст ───────────────────────────────────────────────────────────
test_texts = [
    "Hello, world! This is a test sentence.",
    "ONNX Runtime is a cross-platform inference accelerator.",
]

print()
for text in test_texts:
    print(f"Текст: '{text[:50]}...' " if len(text) > 50 else f"Текст: '{text}'")

    enc = tokenizer(
        text,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    token_type_ids = enc["token_type_ids"].to(DEVICE)

    # ── PyTorch forward pass ──────────────────────────────────────────────────
    with torch.no_grad():
        pt_out = pt_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    pt_hidden = pt_out.last_hidden_state.cpu().numpy()
    pt_pooled = pt_out.pooler_output.cpu().numpy()

    # ── ONNX Runtime forward pass ─────────────────────────────────────────────
    # ORT принимает numpy arrays, не torch тензоры
    ort_inputs = {
        "input_ids":      input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
        "token_type_ids": token_type_ids.cpu().numpy(),
    }
    ort_hidden, ort_pooled = ort_session.run(None, ort_inputs)
    # run(None, ...) — None означает "верни все выходы"

    # ── Сравнение ─────────────────────────────────────────────────────────────
    hidden_ok = np.allclose(pt_hidden, ort_hidden, atol=1e-4, rtol=1e-3)
    pooled_ok = np.allclose(pt_pooled, ort_pooled, atol=1e-4, rtol=1e-3)

    max_diff_hidden = float(np.abs(pt_hidden - ort_hidden).max())
    max_diff_pooled = float(np.abs(pt_pooled - ort_pooled).max())

    status = "OK" if (hidden_ok and pooled_ok) else "FAIL"
    print(f"  [{status}] last_hidden_state: max diff = {max_diff_hidden:.2e}")
    print(f"  [{status}] pooler_output:     max diff = {max_diff_pooled:.2e}")
    print()

print("Валидация завершена. Теперь запускай step3_benchmark.py")
