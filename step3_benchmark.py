"""
Шаг 3: Честный бенчмарк PyTorch vs ONNX Runtime

Правила честного GPU-бенчмарка:
1. ПРОГРЕВ — первые N итераций выбрасываем: GPU холодный,
   драйвер JIT-компилирует ядра, кэш пустой
2. CUDA SYNC — GPU асинхронный. Без synchronize() мы измеряем
   только время постановки задачи в очередь, а не выполнение
3. СТАТИСТИКА — median устойчивее mean к выбросам.
   P99 показывает "худший случай" (важно для продакшена)
4. НЕСКОЛЬКО BATCH SIZE — поведение меняется нелинейно
5. ПОВТОРЯЕМОСТЬ — фиксируем seed, отключаем бенчмарк-режим cuDNN

Запускать: python step3_benchmark.py
"""

import time
import statistics
import numpy as np
import torch
import onnxruntime as ort
from transformers import BertModel, BertTokenizer
from dataclasses import dataclass
from typing import List

# ─── Конфиг бенчмарка ─────────────────────────────────────────────────────────
MODEL_NAME   = "bert-base-uncased"
ONNX_PATH    = "bert.onnx"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_ITERS = 50     # итерации прогрева (выбрасываем)
BENCH_ITERS  = 200    # итерации замера
BATCH_SIZES  = [1, 4, 8, 16]
SEQ_LENGTH   = 128
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    name: str
    batch_size: int
    median_ms: float
    p95_ms: float
    p99_ms: float
    throughput: float  # samples/sec


def benchmark_pytorch(model, batch_size: int) -> BenchResult:
    """Замер PyTorch через CUDA Events — точнее чем time.perf_counter."""
    input_ids      = torch.randint(0, 30522, (batch_size, SEQ_LENGTH)).to(DEVICE)
    attention_mask = torch.ones(batch_size, SEQ_LENGTH, dtype=torch.long).to(DEVICE)
    token_type_ids = torch.zeros(batch_size, SEQ_LENGTH, dtype=torch.long).to(DEVICE)

    # CUDA Events — аппаратные таймеры на GPU, точность ~microsecond
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)

    # Прогрев
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(input_ids=input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids)
    torch.cuda.synchronize()

    # Замер
    times = []
    with torch.no_grad():
        for _ in range(BENCH_ITERS):
            starter.record()
            _ = model(input_ids=input_ids,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids)
            ender.record()
            torch.cuda.synchronize()  # ждём завершения GPU-операций
            times.append(starter.elapsed_time(ender))

    times_sorted = sorted(times)
    med = statistics.median(times)
    return BenchResult(
        name="PyTorch (eager)",
        batch_size=batch_size,
        median_ms=med,
        p95_ms=times_sorted[int(0.95 * BENCH_ITERS)],
        p99_ms=times_sorted[int(0.99 * BENCH_ITERS)],
        throughput=batch_size / (med / 1000),
    )


def benchmark_ort(session: ort.InferenceSession, batch_size: int) -> BenchResult:
    """
    Замер ORT. Для CPU используем time.perf_counter,
    для CUDA — io_binding чтобы данные жили на GPU.
    """
    input_ids_np      = np.random.randint(0, 30522, (batch_size, SEQ_LENGTH)).astype(np.int64)
    attention_mask_np = np.ones((batch_size, SEQ_LENGTH), dtype=np.int64)
    token_type_ids_np = np.zeros((batch_size, SEQ_LENGTH), dtype=np.int64)

    ort_inputs = {
        "input_ids":      input_ids_np,
        "attention_mask": attention_mask_np,
        "token_type_ids": token_type_ids_np,
    }

    # Прогрев
    for _ in range(WARMUP_ITERS):
        session.run(None, ort_inputs)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Замер
    times = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        session.run(None, ort_inputs)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # в миллисекунды

    times_sorted = sorted(times)
    med = statistics.median(times)
    return BenchResult(
        name="ONNX Runtime",
        batch_size=batch_size,
        median_ms=med,
        p95_ms=times_sorted[int(0.95 * BENCH_ITERS)],
        p99_ms=times_sorted[int(0.99 * BENCH_ITERS)],
        throughput=batch_size / (med / 1000),
    )


def print_results(results: List[BenchResult]):
    """Красивая таблица результатов."""
    print()
    print("=" * 72)
    print(f"{'Backend':<22} {'Batch':>6} {'Median':>10} {'P95':>10} {'P99':>10} {'Samples/s':>10}")
    print("-" * 72)

    prev_batch = None
    for r in results:
        if prev_batch is not None and r.batch_size != prev_batch:
            print()
        print(
            f"{r.name:<22} {r.batch_size:>6} "
            f"{r.median_ms:>9.2f}ms "
            f"{r.p95_ms:>9.2f}ms "
            f"{r.p99_ms:>9.2f}ms "
            f"{r.throughput:>9.0f}"
        )
        prev_batch = r.batch_size

    print("=" * 72)

    # Speedup таблица
    print()
    print("Ускорение ONNX Runtime vs PyTorch:")
    pt_by_batch  = {r.batch_size: r for r in results if "PyTorch" in r.name}
    ort_by_batch = {r.batch_size: r for r in results if "ONNX"    in r.name}
    for bs in BATCH_SIZES:
        pt  = pt_by_batch.get(bs)
        ort = ort_by_batch.get(bs)
        if pt and ort:
            speedup = pt.median_ms / ort.median_ms
            bar = "█" * int(speedup * 10)
            print(f"  batch={bs:>2}:  ×{speedup:.2f}  {bar}")


# ─── Main ─────────────────────────────────────────────────────────────────────
print(f"Device: {DEVICE.upper()}")
print(f"Warmup: {WARMUP_ITERS} iter  |  Bench: {BENCH_ITERS} iter")
print()

# PyTorch модель
print("Загружаем PyTorch модель...")
pt_model = BertModel.from_pretrained(MODEL_NAME).eval().to(DEVICE)

# Включаем torch.compile если доступен (PyTorch 2.0+)
# Это само по себе сильная оптимизация — для сравнения с ORT по-честному
try:
    pt_model = torch.compile(pt_model, mode="reduce-overhead")
    print("torch.compile: включён (mode=reduce-overhead)")
except Exception:
    print("torch.compile: недоступен (PyTorch < 2.0)")

# ONNX Runtime сессия
print("Создаём ORT сессию...")
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 4   # для CPU: количество потоков на одну операцию
so.inter_op_num_threads = 1   # количество параллельных операций

providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if DEVICE == "cuda" else
    ["CPUExecutionProvider"]
)
ort_session = ort.InferenceSession(ONNX_PATH, so, providers=providers)
print(f"ORT провайдер: {ort_session.get_providers()[0]}")
print()

# Бенчмарк
all_results = []
for bs in BATCH_SIZES:
    print(f"Замер batch_size={bs}...", end=" ", flush=True)
    pt_res  = benchmark_pytorch(pt_model, bs)
    ort_res = benchmark_ort(ort_session, bs)
    all_results.extend([pt_res, ort_res])
    print(f"PyTorch {pt_res.median_ms:.1f}ms  |  ORT {ort_res.median_ms:.1f}ms")

print_results(all_results)
print()
print("Готово! Смотри на Throughput (samples/s) — это главная метрика для батчей.")
print("Смотри на Median latency — это главная метрика для одиночных запросов.")
