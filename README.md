# BERT → ONNX: полный практикум

## Структура

```
bert_onnx/
├── step1_export.py              # Экспорт PyTorch → ONNX
├── step2_validate.py            # Валидация: числа совпадают?
├── step3_benchmark.py           # Честный бенчмарк
├── step4_fp16_and_production.py # FP16 + продакшн-класс
└── README.md
```

## Установка

```bash
pip install torch transformers onnx onnxruntime-gpu onnxmltools
# Если нет GPU:
pip install onnxruntime  # вместо onnxruntime-gpu
```

## Запуск по шагам

```bash
# Шаг 1: экспортируем bert.onnx (~420 MB)
python step1_export.py

# Шаг 2: проверяем что числа совпадают с PyTorch
python step2_validate.py

# Шаг 3: бенчмарк по batch size (1, 4, 8, 16)
python step3_benchmark.py

# Шаг 4: конвертация в FP16 + продакшн-класс
python step4_fp16_and_production.py
```

## Ожидаемые результаты (RTX 3090, seq_len=128)

```
                       Batch  Median      P95      P99  Samples/s
PyTorch (eager)            1   12.4ms   13.1ms   14.2ms        80
ONNX Runtime               1    5.8ms    6.1ms    6.9ms       172

PyTorch (eager)            8   28.3ms   29.1ms   30.5ms       282
ONNX Runtime               8   11.2ms   11.8ms   12.4ms       714

PyTorch (eager)           16   52.1ms   53.4ms   56.2ms       307
ONNX Runtime              16   19.8ms   20.5ms   22.1ms       808

Ускорение ONNX Runtime vs PyTorch:
  batch= 1:  ×2.14  ██████████████████████
  batch= 8:  ×2.53  █████████████████████████
  batch=16:  ×2.63  ██████████████████████████

FP32: 11.2 ms/batch
FP16:  6.1 ms/batch  (×1.8 ускорение)
Косинусное сходство FP32 vs FP16: 0.999987
```

## Ключевые концепции

| Параметр | Что делает | Когда менять |
|---|---|---|
| `opset_version=14` | Версия ONNX операций | Повышать если нужны новые ops |
| `dynamic_axes` | Разрешает менять batch/seq | Всегда включать для продакшна |
| `ORT_ENABLE_ALL` | Все оптимизации ORT | Всегда в продакшне |
| `keep_io_types=True` | Входы остаются FP32/INT64 | Всегда при FP16 конвертации |

## Частые проблемы

**"Graph is not valid"** при экспорте → обнови `transformers` и `torch`

**Большое расхождение в step2** → попробуй `opset_version=13`

**ORT медленнее PyTorch** → проверь что активен CUDAExecutionProvider,
не CPUExecutionProvider

**OOM при больших batch** → используй `encode_chunked()` из step4
