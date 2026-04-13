"""
Шаг 4: FP16 оптимизация + продакшн-обёртка

onnxmltools.convert_float_to_float16 проходит по всему графу
и заменяет FP32 операции на FP16 там, где это безопасно.

Это даёт дополнительные x1.5–2× на современных GPU
(Ampere, Ada Lovelace) без потери качества для большинства моделей.
"""

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxmltools.utils.float16_converter import convert_float_to_float16
from transformers import BertTokenizer

# ─── Конфиг ───────────────────────────────────────────────────────────────────
ONNX_FP32_PATH = "bert.onnx"
ONNX_FP16_PATH = "bert_fp16.onnx"
MODEL_NAME     = "bert-base-uncased"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────


# ── FP16 конвертация ──────────────────────────────────────────────────────────
def convert_to_fp16(fp32_path: str, fp16_path: str):
    print(f"Конвертируем {fp32_path} → {fp16_path}...")
    model_fp32 = onnx.load(fp32_path)

    # keep_io_types=True — входы/выходы остаются INT64/FP32
    # чтобы не менять интерфейс сессии
    model_fp16 = convert_float_to_float16(
        model_fp32,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    onnx.save(model_fp16, fp16_path)

    import os
    orig_mb = os.path.getsize(fp32_path) / 1024 / 1024
    fp16_mb = os.path.getsize(fp16_path) / 1024 / 1024
    print(f"  FP32: {orig_mb:.1f} MB  →  FP16: {fp16_mb:.1f} MB  ({fp16_mb/orig_mb*100:.0f}%)")


# ── Продакшн-класс для инференса ──────────────────────────────────────────────
class BertInferenceSession:
    """
    Готовая к продакшну обёртка вокруг ONNX Runtime.

    Особенности:
    - Принимает сырой текст или список текстов
    - Автоматически паддит до нужной длины
    - Возвращает эмбеддинги как numpy arrays
    - Поддерживает CPU и GPU без изменений кода

    Пример использования:
        session = BertInferenceSession("bert_fp16.onnx")
        emb = session.encode(["Hello world", "Another text"])
        print(emb.shape)  # (2, 768)
    """

    def __init__(
        self,
        onnx_path: str,
        model_name: str = "bert-base-uncased",
        max_seq_len: int = 128,
        use_gpu: bool = True,
    ):
        self.tokenizer  = BertTokenizer.from_pretrained(model_name)
        self.max_seq_len = max_seq_len

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Логируем предупреждения, но не info — иначе слишком шумно
        so.log_severity_level = 2

        # Выбираем провайдер
        if use_gpu and torch.cuda.is_available():
            # CUDAExecutionProvider можно настраивать:
            cuda_opts = {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4 GB лимит
                "cudnn_conv_algo_search": "EXHAUSTIVE",    # перебирает лучший алгоритм
            }
            providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._session  = ort.InferenceSession(onnx_path, so, providers=providers)
        self._provider = self._session.get_providers()[0]
        print(f"BertInferenceSession: {onnx_path} | провайдер: {self._provider}")

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Возвращает CLS-токен эмбеддинги: shape (len(texts), 768).
        CLS pooler_output — стандартный способ получить
        sentence-level представление из BERT.
        """
        enc = self.tokenizer(
            texts,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",  # сразу numpy, без torch
        )

        outputs = self._session.run(
            ["pooler_output"],  # запрашиваем только нужный выход
            {
                "input_ids":      enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
                "token_type_ids": enc["token_type_ids"].astype(np.int64),
            },
        )
        return outputs[0]  # shape: (batch, 768)

    def encode_chunked(self, texts: list[str], chunk_size: int = 32) -> np.ndarray:
        """
        Для больших списков текстов: разбиваем на чанки
        чтобы не переполнить GPU память.
        """
        results = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            results.append(self.encode(chunk))
        return np.concatenate(results, axis=0)


# ─── Демо ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    # Конвертируем в FP16
    convert_to_fp16(ONNX_FP32_PATH, ONNX_FP16_PATH)
    print()

    # Сравниваем FP32 и FP16 сессии
    session_fp32 = BertInferenceSession(ONNX_FP32_PATH)
    session_fp16 = BertInferenceSession(ONNX_FP16_PATH)
    print()

    test_texts = [
        "Machine learning is transforming the world.",
        "ONNX Runtime provides cross-platform inference.",
        "FP16 precision halves memory bandwidth usage.",
        "Operator fusion reduces GPU kernel launch overhead.",
    ]

    # Прогрев
    for _ in range(20):
        session_fp32.encode(test_texts)
        session_fp16.encode(test_texts)

    # Замер FP32
    t0 = time.perf_counter()
    for _ in range(200):
        session_fp32.encode(test_texts)
    fp32_ms = (time.perf_counter() - t0) / 200 * 1000

    # Замер FP16
    t0 = time.perf_counter()
    for _ in range(200):
        session_fp16.encode(test_texts)
    fp16_ms = (time.perf_counter() - t0) / 200 * 1000

    print(f"FP32: {fp32_ms:.2f} ms/batch")
    print(f"FP16: {fp16_ms:.2f} ms/batch  (×{fp32_ms/fp16_ms:.1f} ускорение)")

    # Проверка качества: эмбеддинги должны быть близки
    emb_fp32 = session_fp32.encode(test_texts[:1])
    emb_fp16 = session_fp16.encode(test_texts[:1])
    cosine   = float(
        np.dot(emb_fp32[0], emb_fp16[0]) /
        (np.linalg.norm(emb_fp32[0]) * np.linalg.norm(emb_fp16[0]))
    )
    print(f"\nКосинусное сходство FP32 vs FP16: {cosine:.6f}")
    print("(1.0 = идентичны, >0.9999 = отлично для практики)")
    print()
    print("Используй BertInferenceSession в своём сервисе как drop-in замену!")
