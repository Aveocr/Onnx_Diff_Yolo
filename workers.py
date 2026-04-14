import sys
import os
import time
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

from logging_setup import logger
from constants import FORMAT_CONFIGS, EXPORT_SUFFIXES
from utils import _check_yaml_paths


# ─── DepsInstallThread ───────────────────────────────────────────────────────

class DepsInstallThread(QThread):
    """Устанавливает pip-пакеты по одному, стримит вывод в лог."""
    log_signal      = pyqtSignal(str)
    finished_signal = pyqtSignal()

    # CPU-вариант: onnxruntime без GPU
    PACKAGES_CPU = ['onnx', 'onnxruntime', 'onnxslim']
    # GPU-вариант: onnxruntime-gpu заменяет onnxruntime, tensorrt добавляется
    PACKAGES_GPU = ['onnx', 'onnxruntime-gpu', 'onnxslim', 'tensorrt']

    def __init__(self, packages: list):
        super().__init__()
        self.packages = packages

    def run(self):
        import subprocess
        try:
            for pkg in self.packages:
                self.log_signal.emit(f"pip install --upgrade {pkg} ...")
                proc = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', '--upgrade', pkg],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                )
                if proc.returncode == 0:
                    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
                    summary = lines[-1] if lines else "OK"
                    self.log_signal.emit(f"  {summary}")
                else:
                    lines = [l for l in proc.stderr.strip().splitlines() if l.strip()]
                    err = lines[-1] if lines else "неизвестная ошибка"
                    self.log_signal.emit(f"  Ошибка: {err}")
                    logger.error("pip install %s: %s", pkg, proc.stderr)
        except Exception as e:
            logger.exception("DepsInstallThread")
            self.log_signal.emit(f"Ошибка установки: {e}")
        finally:
            self.log_signal.emit("Установка завершена. Перезапустите приложение.")
            self.finished_signal.emit()


# ─── BenchmarkThread ──────────────────────────────────────────────────────────

class BenchmarkThread(QThread):
    """Поток для валидации одной модели."""
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)  # всегда эмитируется, {} при ошибке

    def __init__(self, model_path, data_yaml, device, imgsz, half, batch_size, label=""):
        super().__init__()
        self.model_path = model_path
        self.data_yaml  = data_yaml
        self.device     = device
        self.imgsz      = imgsz
        self.half       = half
        self.batch_size = batch_size
        self.label      = label

    def run(self):
        try:
            self.log_signal.emit(f"Бенчмарк [{self.label}]: {self.model_path}")
            self.log_signal.emit(f"  Датасет: {self.data_yaml}  |  {self.device}, imgsz={self.imgsz}, batch={self.batch_size}")

            model      = YOLO(self.model_path)
            model_type = "ONNX" if self.model_path.endswith('.onnx') else \
                         "TRT"  if self.model_path.endswith('.engine') else "PyTorch"
            self.log_signal.emit(f"  Тип модели: {model_type}")

            start   = time.perf_counter()
            metrics = model.val(
                data=self.data_yaml,
                imgsz=self.imgsz,
                batch=self.batch_size,
                device=self.device,
                half=self.half,
                verbose=False,
                plots=False,
                workers=0,  # предотвращает multiprocessing spawn в потоке на Windows
            )
            val_time = time.perf_counter() - start

            size_mb = os.path.getsize(self.model_path) / 1024 / 1024
            self.log_signal.emit(f"  Готово за {val_time:.2f} сек.  Размер: {size_mb:.1f} МБ")
            self.finished_signal.emit({
                'label':        self.label,
                'model_path':   self.model_path,
                'model_type':   model_type,
                'val_time':     val_time,
                'map50':        float(metrics.box.map50),
                'map50_95':     float(metrics.box.map),
                'fitness':      float(metrics.fitness),
                'file_size_mb': size_mb,
            })
        except Exception as e:
            logger.exception("BenchmarkThread [%s]", self.label)
            self.log_signal.emit(f"  Ошибка бенчмарка [{self.label}]: {e}")
            self.finished_signal.emit({})  # всегда эмитируем
        finally:
            self.progress_signal.emit(100)


# ─── ExportThread ─────────────────────────────────────────────────────────────

class ExportThread(QThread):
    """Поток для экспорта модели в один формат."""
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)  # путь к файлу или "" при ошибке

    def __init__(self, model_path, fmt_label, fmt_config, imgsz, opset, dynamic, simplify, data_yaml):
        super().__init__()
        self.model_path = model_path
        self.fmt_label  = fmt_label
        self.fmt_config = fmt_config
        self.imgsz      = imgsz
        self.opset      = opset
        self.dynamic    = dynamic
        self.simplify   = simplify
        self.data_yaml  = data_yaml

    def run(self):
        try:
            self.log_signal.emit(f"Загрузка модели для экспорта [{self.fmt_label}]...")
            self.progress_signal.emit(20)
            model = YOLO(self.model_path)

            self.log_signal.emit(f"Экспорт в {self.fmt_label}...")
            self.progress_signal.emit(40)

            kwargs = dict(
                format=self.fmt_config['format'],
                imgsz=self.imgsz,
                half=self.fmt_config['half'],
                int8=self.fmt_config['int8'],
                dynamic=self.dynamic,
                simplify=self.simplify,
                opset=self.opset,
            )
            if self.fmt_config['int8']:
                kwargs['data'] = self.data_yaml

            export_path = model.export(**kwargs)

            self.progress_signal.emit(90)
            self.log_signal.emit(f"Экспорт [{self.fmt_label}] завершён: {export_path}")
            self.finished_signal.emit(str(export_path))
        except Exception as e:
            logger.exception("ExportThread [%s]", self.fmt_label)
            self.log_signal.emit(f"Ошибка экспорта [{self.fmt_label}]: {e}")
            self.finished_signal.emit("")
        finally:
            self.progress_signal.emit(100)


# ─── RunAllThread ─────────────────────────────────────────────────────────────

class RunAllThread(QThread):
    """Запускает экспорт + бенчмарк для всех выбранных форматов последовательно."""
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    step_signal     = pyqtSignal(str)
    result_signal   = pyqtSignal(str, dict)
    error_signal    = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, pt_path, formats_to_run, data_yaml, device,
                 imgsz, batch, opset, dynamic, simplify, half_pt):
        super().__init__()
        self.pt_path        = pt_path
        self.formats_to_run = formats_to_run
        self.data_yaml      = data_yaml
        self.device         = device
        self.imgsz          = imgsz
        self.batch          = batch
        self.opset          = opset
        self.dynamic        = dynamic
        self.simplify       = simplify
        self.half_pt        = half_pt

    def run(self):
        try:
            total   = len(self.formats_to_run) * 2
            current = 0

            for label in self.formats_to_run:
                cfg = FORMAT_CONFIGS[label]

                # ── Экспорт ──────────────────────────────────────────────────────
                if label == 'PyTorch':
                    model_path = self.pt_path
                    current += 1
                else:
                    self.step_signal.emit(f"Экспорт {label}...")
                    model_path = self._do_export(label, cfg)
                    current += 1
                    self.progress_signal.emit(int(current / total * 100))

                    if not model_path:
                        self.error_signal.emit(f"Экспорт {label} не удался — пропускаем")
                        current += 1  # пропускаем и шаг бенчмарка
                        self.progress_signal.emit(int(current / total * 100))
                        continue

                # ── Бенчмарк ─────────────────────────────────────────────────────
                self.step_signal.emit(f"Бенчмарк {label}...")
                half = self.half_pt if label == 'PyTorch' else cfg['half']
                results = self._do_benchmark(model_path, label, half)
                current += 1
                self.progress_signal.emit(int(current / total * 100))

                if results:
                    self.result_signal.emit(label, results)
                else:
                    self.error_signal.emit(f"Бенчмарк {label} не удался")

            self.step_signal.emit("Готово")
        except Exception as e:
            logger.exception("RunAllThread.run: неперехваченная ошибка")
            self.log_signal.emit(f"Критическая ошибка потока: {e}")
            self.step_signal.emit("Ошибка")
        finally:
            self.finished_signal.emit()

    def _do_export(self, label, cfg):
        try:
            self.log_signal.emit(f"[Экспорт] {label}...")

            if cfg.get('post_quant'):
                return self._do_export_onnx_int8()

            is_trt = cfg['format'] == 'engine'

            # TRT (C++) не работает с не-ASCII символами в пути (например, кириллица).
            # Если путь содержит не-ASCII — копируем .pt во временную папку, экспортируем там.
            if is_trt and not self.pt_path.isascii():
                return self._do_export_trt_via_tempdir(label, cfg)

            model  = YOLO(self.pt_path)
            # TRT не поддерживает динамические оси так же, как ONNX:
            # dynamic=True в ONNX-модели вызывает "failed to load ONNX file" в парсере TRT
            kwargs = dict(
                format=cfg['format'],
                imgsz=self.imgsz,
                half=cfg['half'],
                dynamic=False if is_trt else self.dynamic,
                simplify=self.simplify,
                opset=self.opset,
            )
            # int8=True только для TRT — Ultralytics < 8.6 не поддерживает его для ONNX
            if cfg['int8']:
                kwargs['int8'] = True
                kwargs['data'] = self.data_yaml

            result = model.export(**kwargs)
            if result is None:
                self.log_signal.emit(f"  Экспорт [{label}] вернул None — возможно, не установлен onnx")
                return ""
            raw_path = Path(str(result))
            if not raw_path.exists():
                self.log_signal.emit(f"  Файл экспорта не найден: {raw_path}")
                return ""
            dest = self._copy_to_onnx_dir(raw_path, label)
            self.log_signal.emit(f"  Сохранён: {dest}")
            return str(dest)
        except AssertionError as e:
            self.log_signal.emit(f"  Параметр не поддерживается текущей версией Ultralytics: {e}")
            return ""
        except Exception as e:
            msg = str(e)
            trt_keywords = ('tensorrt', 'engine', 'failed to load onnx', 'onnx parser')
            if cfg.get('format') == 'engine' or any(k in msg.lower() for k in trt_keywords):
                logger.warning("RunAllThread._do_export TRT [%s]: %s", label, msg)
                self.log_signal.emit(f"  TensorRT недоступен или несовместим: {msg}")
            else:
                logger.exception("RunAllThread._do_export [%s]", label)
                self.log_signal.emit(f"  Ошибка: {msg}")
            return ""

    def _do_export_trt_via_tempdir(self, label, cfg):
        """Экспорт TRT через временную папку с ASCII-путём.
        TensorRT — C++ библиотека, не поддерживает кириллицу и другие не-ASCII символы в пути."""
        import shutil
        import tempfile

        pt_name = Path(self.pt_path).name  # yolo26s.pt
        self.log_signal.emit(f"  Путь содержит не-ASCII символы — копируем модель во временную папку...")

        with tempfile.TemporaryDirectory(prefix='trt_export_') as tmp:
            tmp_pt = Path(tmp) / pt_name
            shutil.copy2(self.pt_path, tmp_pt)
            self.log_signal.emit(f"  Временный путь: {tmp_pt}")

            try:
                model  = YOLO(str(tmp_pt))
                kwargs = dict(
                    format=cfg['format'],
                    imgsz=self.imgsz,
                    half=cfg['half'],
                    dynamic=False,   # TRT не поддерживает dynamic axes
                    simplify=self.simplify,
                    opset=self.opset,
                )
                if cfg['int8']:
                    kwargs['int8'] = True
                    kwargs['data'] = self.data_yaml

                result = model.export(**kwargs)
                if result is None:
                    self.log_signal.emit(f"  Экспорт [{label}] вернул None")
                    return ""

                raw_path = Path(str(result))
                if not raw_path.exists():
                    self.log_signal.emit(f"  Файл экспорта не найден: {raw_path}")
                    return ""

                dest = self._copy_to_onnx_dir(raw_path, label)
                self.log_signal.emit(f"  Сохранён: {dest}")
                return str(dest)
            except Exception as e:
                msg = str(e)
                logger.warning("RunAllThread._do_export_trt_via_tempdir [%s]: %s", label, msg)
                self.log_signal.emit(f"  TensorRT недоступен или несовместим: {msg}")
                return ""

    def _do_export_onnx_int8(self):
        """INT8-квантизация через onnxruntime.quantization (работает со всеми версиями Ultralytics)."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            self.log_signal.emit("  onnxruntime не установлен. Установите: pip install onnxruntime")
            return ""

        try:
            # Шаг 1: экспортируем FP32-ONNX как базу
            self.log_signal.emit("  Шаг 1/2: экспорт FP32 ONNX для квантизации...")
            model    = YOLO(self.pt_path)
            fp32_raw = Path(str(model.export(
                format='onnx', imgsz=self.imgsz,
                half=False, dynamic=self.dynamic,
                simplify=self.simplify, opset=self.opset,
            )))

            # Шаг 2: квантизация в INT8
            self.log_signal.emit("  Шаг 2/2: квантизация INT8...")
            pt_stem  = Path(self.pt_path).stem
            onnx_dir = Path(self.pt_path).parent / '.onnx'
            onnx_dir.mkdir(exist_ok=True)
            dest = onnx_dir / f"{pt_stem}_int8.onnx"

            quantize_dynamic(
                str(fp32_raw),
                str(dest),
                weight_type=QuantType.QInt8,
            )
            self.log_signal.emit(f"  Сохранён: {dest}")
            return str(dest)
        except Exception as e:
            logger.exception("_do_export_onnx_int8")
            self.log_signal.emit(f"  Ошибка INT8-квантизации: {e}")
            return ""

    def _copy_to_onnx_dir(self, raw_path: Path, label: str) -> Path:
        """Копирует экспортированный файл в {pt_dir}/.onnx/ с именем yolov8n_fp16.onnx."""
        import shutil
        pt_stem  = Path(self.pt_path).stem                    # e.g. "yolov8n"
        suffix, ext = EXPORT_SUFFIXES.get(label, ('', raw_path.suffix))
        dest_name = f"{pt_stem}{suffix}{ext}"                 # e.g. "yolov8n_fp16.onnx"

        onnx_dir = Path(self.pt_path).parent / '.onnx'
        onnx_dir.mkdir(exist_ok=True)
        dest = onnx_dir / dest_name

        if raw_path.resolve() != dest.resolve():
            shutil.copy2(raw_path, dest)
        return dest

    def _do_benchmark(self, model_path, label, half):
        try:
            self.log_signal.emit(f"[Бенчмарк] {label}: {model_path}")

            # Проверяем, что путь датасета из yaml существует на этой машине
            err = _check_yaml_paths(self.data_yaml)
            if err:
                self.log_signal.emit(f"  Ошибка датасета: {err}")
                return {}

            model   = YOLO(model_path)
            start   = time.perf_counter()
            metrics = model.val(
                data=self.data_yaml,
                imgsz=self.imgsz,
                batch=self.batch,
                device=self.device,
                half=half,
                verbose=False,
                plots=False,
                workers=0,  # предотвращает multiprocessing spawn в потоке на Windows
            )
            val_time = time.perf_counter() - start
            size_mb = os.path.getsize(model_path) / 1024 / 1024
            self.log_signal.emit(f"  mAP50={metrics.box.map50:.4f}  time={val_time:.1f}s  size={size_mb:.1f}MB")
            return {
                'label':       label,
                'model_path':  model_path,
                'val_time':    val_time,
                'map50':       float(metrics.box.map50),
                'map50_95':    float(metrics.box.map),
                'fitness':     float(metrics.fitness),
                'file_size_mb': size_mb,
            }
        except (ImportError, OSError) as e:
            msg = str(e).lower()
            if 'onnxruntime' in msg or 'dll' in msg or 'load failed' in msg:
                logger.error("onnxruntime недоступен [%s]: %s", label, e)
                self.log_signal.emit(
                    f"  onnxruntime не загрузился (DLL error).\n"
                    f"  Запустите: pip install --upgrade onnxruntime\n"
                    f"  Или установите Visual C++ Redistributable 2022 x64."
                )
            else:
                logger.exception("RunAllThread._do_benchmark [%s]", label)
                self.log_signal.emit(f"  Ошибка импорта: {e}")
            return {}
        except Exception as e:
            logger.exception("RunAllThread._do_benchmark [%s]", label)
            self.log_signal.emit(f"  Ошибка: {e}")
            return {}
