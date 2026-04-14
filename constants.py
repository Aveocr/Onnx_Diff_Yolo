try:
    import torch
    from PyQt5.QtGui import QColor
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Установите зависимости: pip install PyQt5 ultralytics torch")
    import sys
    sys.exit(1)


CUDA_AVAILABLE = torch.cuda.is_available()

FORMAT_CONFIGS = {
    'PyTorch':   {'format': None,     'half': False, 'int8': False, 'post_quant': False, 'needs_cuda': False},
    'ONNX FP32': {'format': 'onnx',   'half': False, 'int8': False, 'post_quant': False, 'needs_cuda': False},
    'ONNX FP16': {'format': 'onnx',   'half': True,  'int8': False, 'post_quant': False, 'needs_cuda': False},
    # INT8 для ONNX делается через onnxruntime.quantization после экспорта FP32
    # (Ultralytics не поддерживает int8=True для format='onnx' в версиях < 8.6)
    'ONNX INT8': {'format': 'onnx',   'half': False, 'int8': False, 'post_quant': True,  'needs_cuda': False},
    'TRT FP16':  {'format': 'engine', 'half': True,  'int8': False, 'post_quant': False, 'needs_cuda': True},
    'TRT INT8':  {'format': 'engine', 'half': False, 'int8': True,  'post_quant': False, 'needs_cuda': True},
}

TABLE_HEADERS = ["Формат", "mAP50", "mAP50-95", "Время(с)", "Δ mAP50 vs PT", "Скорость vs PT", "Размер (МБ)", "Размер vs PT"]

# Суффиксы для имён файлов экспортированных моделей
EXPORT_SUFFIXES = {
    'ONNX FP32': ('_fp32',     '.onnx'),
    'ONNX FP16': ('_fp16',     '.onnx'),
    'ONNX INT8': ('_int8',     '.onnx'),
    'TRT FP16':  ('_trt_fp16', '.engine'),
    'TRT INT8':  ('_trt_int8', '.engine'),
}

COLOR_GREEN = QColor(200, 240, 200)
COLOR_RED   = QColor(255, 200, 200)
COLOR_GRAY  = QColor(230, 230, 230)
