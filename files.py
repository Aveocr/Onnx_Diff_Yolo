import sys
import os
import glob
import json
import csv
import time
import traceback
from datetime import datetime
from pathlib import Path

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
        QGroupBox, QComboBox, QLineEdit, QCheckBox, QMessageBox,
        QTableWidget, QTableWidgetItem, QHeaderView
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QTextCursor, QColor
    from ultralytics import YOLO
    import torch
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Установите зависимости: pip install PyQt5 ultralytics torch")
    sys.exit(1)

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


# ─── Константы ────────────────────────────────────────────────────────────────

CUDA_AVAILABLE = torch.cuda.is_available()

FORMAT_CONFIGS = {
    'PyTorch':   {'format': None,     'half': False, 'int8': False, 'needs_cuda': False},
    'ONNX FP32': {'format': 'onnx',   'half': False, 'int8': False, 'needs_cuda': False},
    'ONNX FP16': {'format': 'onnx',   'half': True,  'int8': False, 'needs_cuda': False},
    'ONNX INT8': {'format': 'onnx',   'half': False, 'int8': True,  'needs_cuda': False},
    'TRT FP16':  {'format': 'engine', 'half': True,  'int8': False, 'needs_cuda': True},
    'TRT INT8':  {'format': 'engine', 'half': False, 'int8': True,  'needs_cuda': True},
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


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _parse_int(line_edit, default):
    try:
        return int(line_edit.text())
    except ValueError:
        return default


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
            self.log_signal.emit(f"  Ошибка бенчмарка [{self.label}]: {e}")
            traceback.print_exc()
            self.finished_signal.emit({})  # Bug 1 fix: всегда эмитируем
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
            self.log_signal.emit(f"Ошибка экспорта [{self.fmt_label}]: {e}")
            traceback.print_exc()
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
        self.finished_signal.emit()

    def _do_export(self, label, cfg):
        try:
            self.log_signal.emit(f"[Экспорт] {label}...")
            model  = YOLO(self.pt_path)
            kwargs = dict(
                format=cfg['format'],
                imgsz=self.imgsz,
                half=cfg['half'],
                int8=cfg['int8'],
                dynamic=self.dynamic,
                simplify=self.simplify,
                opset=self.opset,
            )
            if cfg['int8']:
                kwargs['data'] = self.data_yaml
            raw_path = Path(str(model.export(**kwargs)))

            # Копируем в .onnx/ рядом с исходным .pt под человекочитаемым именем
            dest = self._copy_to_onnx_dir(raw_path, label)
            self.log_signal.emit(f"  Сохранён: {dest}")
            return str(dest)
        except Exception as e:
            msg = str(e)
            if 'tensorrt' in msg.lower() or 'engine' in msg.lower():
                self.log_signal.emit(f"  TensorRT недоступен: {msg}")
            else:
                self.log_signal.emit(f"  Ошибка: {msg}")
            traceback.print_exc()
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
        except Exception as e:
            self.log_signal.emit(f"  Ошибка: {e}")
            traceback.print_exc()
            return {}


# ─── YOLOBenchmarkApp ─────────────────────────────────────────────────────────

class YOLOBenchmarkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Export & Benchmark Tool")
        self.setGeometry(100, 100, 980, 780)

        self.export_thread    = None
        self.benchmark_thread = None
        self.run_all_thread   = None

        self.format_checkboxes = {}
        self.results_store     = {}
        self.exported_onnx_path = None

        self.init_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(6)

        title = QLabel("Инструмент экспорта и сравнения моделей YOLO")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 8px;")
        layout.addWidget(title)

        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_dataset_group())
        layout.addWidget(self._build_formats_group())
        layout.addWidget(self._build_run_group())
        layout.addWidget(self._build_results_group())
        layout.addWidget(self._build_log_group())

        self.update_gpu_info()

    def _build_model_group(self):
        group  = QGroupBox("1. Выбор модели (.pt)")
        layout = QVBoxLayout()

        row = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Путь к файлу модели (.pt)")
        btn = QPushButton("Обзор")
        btn.clicked.connect(self.browse_model)
        row.addWidget(QLabel("Модель:"))
        row.addWidget(self.model_path_edit)
        row.addWidget(btn)
        layout.addLayout(row)

        self.model_info_label = QLabel("Выберите файл модели")
        self.model_info_label.setStyleSheet("color: gray;")
        layout.addWidget(self.model_info_label)

        group.setLayout(layout)
        return group

    def _build_dataset_group(self):
        group  = QGroupBox("2. Датасет")
        layout = QVBoxLayout()

        row = QHBoxLayout()
        self.data_yaml_edit = QLineEdit()
        self.data_yaml_edit.setPlaceholderText("Путь к data.yaml")
        folder_btn = QPushButton("Выбрать папку")
        folder_btn.clicked.connect(self.browse_dataset_folder)
        yaml_btn = QPushButton("Выбрать data.yaml")
        yaml_btn.clicked.connect(self.browse_data_yaml)
        row.addWidget(QLabel("Датасет:"))
        row.addWidget(self.data_yaml_edit)
        row.addWidget(folder_btn)
        row.addWidget(yaml_btn)
        layout.addLayout(row)

        group.setLayout(layout)
        return group

    def _build_formats_group(self):
        group  = QGroupBox("3. Форматы экспорта и параметры")
        layout = QVBoxLayout()

        # Чекбоксы форматов
        cb_row = QHBoxLayout()
        cb_row.addWidget(QLabel("Форматы:"))
        for label, cfg in FORMAT_CONFIGS.items():
            cb = QCheckBox(label)
            cb.setChecked(label in ('PyTorch', 'ONNX FP32', 'ONNX FP16'))
            if cfg['needs_cuda'] and not CUDA_AVAILABLE:
                cb.setEnabled(False)
                cb.setToolTip("Требуется CUDA и TensorRT")
            self.format_checkboxes[label] = cb
            cb_row.addWidget(cb)
        cb_row.addStretch()
        layout.addLayout(cb_row)

        # Параметры экспорта
        opts_row = QHBoxLayout()
        opts_row.addWidget(QLabel("imgsz:"))
        self.imgsz_edit = QLineEdit("640")
        self.imgsz_edit.setMaximumWidth(70)
        opts_row.addWidget(self.imgsz_edit)

        opts_row.addWidget(QLabel("Opset:"))
        self.opset_edit = QLineEdit("17")
        self.opset_edit.setMaximumWidth(50)
        opts_row.addWidget(self.opset_edit)

        opts_row.addWidget(QLabel("Batch:"))
        self.batch_edit = QLineEdit("16")
        self.batch_edit.setMaximumWidth(50)
        opts_row.addWidget(self.batch_edit)

        opts_row.addWidget(QLabel("Устройство:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("cpu")
        if CUDA_AVAILABLE:
            self.device_combo.addItem("0")
        opts_row.addWidget(self.device_combo)

        self.dynamic_check = QCheckBox("Dynamic")
        self.dynamic_check.setChecked(True)
        opts_row.addWidget(self.dynamic_check)

        self.simplify_check = QCheckBox("Simplify")
        self.simplify_check.setChecked(True)
        opts_row.addWidget(self.simplify_check)

        self.half_pt_check = QCheckBox("FP16 для PyTorch")
        self.half_pt_check.setChecked(CUDA_AVAILABLE)
        self.half_pt_check.setToolTip("Использовать half precision при валидации оригинальной PT модели")
        opts_row.addWidget(self.half_pt_check)

        opts_row.addStretch()
        layout.addLayout(opts_row)

        group.setLayout(layout)
        return group

    def _build_run_group(self):
        group  = QGroupBox("4. Запуск")
        layout = QVBoxLayout()

        self.run_all_btn = QPushButton("▶  Запустить всё")
        self.run_all_btn.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 8px; background-color: #4CAF50; color: white;"
        )
        self.run_all_btn.clicked.connect(self.run_all)
        layout.addWidget(self.run_all_btn)

        self.step_label = QLabel("")
        self.step_label.setStyleSheet("color: #555; font-style: italic;")
        layout.addWidget(self.step_label)

        self.overall_progress = QProgressBar()
        self.overall_progress.setVisible(False)
        layout.addWidget(self.overall_progress)

        # Ручное управление (экспорт + бенчмарк по одному)
        manual_row = QHBoxLayout()
        self.export_btn = QPushButton("Экспорт в ONNX (ручной)")
        self.export_btn.clicked.connect(self.export_model)
        self.run_benchmark_pt_btn = QPushButton("Бенчмарк PyTorch")
        self.run_benchmark_pt_btn.clicked.connect(lambda: self.run_benchmark("pt"))
        self.run_benchmark_onnx_btn = QPushButton("Бенчмарк ONNX")
        self.run_benchmark_onnx_btn.clicked.connect(lambda: self.run_benchmark("onnx"))
        manual_row.addWidget(self.export_btn)
        manual_row.addWidget(self.run_benchmark_pt_btn)
        manual_row.addWidget(self.run_benchmark_onnx_btn)
        layout.addLayout(manual_row)

        self.export_progress = QProgressBar()
        self.export_progress.setVisible(False)
        layout.addWidget(self.export_progress)

        group.setLayout(layout)
        return group

    def _build_results_group(self):
        group  = QGroupBox("5. Результаты сравнения")
        layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(len(TABLE_HEADERS))
        self.results_table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setMinimumHeight(160)
        self.results_table.setMaximumHeight(200)
        layout.addWidget(self.results_table)

        btn_row = QHBoxLayout()
        save_json_btn = QPushButton("Сохранить JSON")
        save_json_btn.clicked.connect(self._save_results_json_dialog)
        save_csv_btn = QPushButton("Сохранить CSV")
        save_csv_btn.clicked.connect(self._save_results_csv_dialog)
        self.autosave_label = QLabel("")
        self.autosave_label.setStyleSheet("color: gray; font-size: 11px;")
        btn_row.addWidget(save_json_btn)
        btn_row.addWidget(save_csv_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.autosave_label)
        layout.addLayout(btn_row)

        group.setLayout(layout)
        return group

    def _build_log_group(self):
        group  = QGroupBox("Лог")
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        self.log_text.setMaximumHeight(180)
        layout.addWidget(self.log_text)

        clear_btn = QPushButton("Очистить лог")
        clear_btn.clicked.connect(self.clear_log)
        layout.addWidget(clear_btn)

        group.setLayout(layout)
        return group

    # ── GPU-инфо ──────────────────────────────────────────────────────────────

    def update_gpu_info(self):
        if CUDA_AVAILABLE:
            self.log(f"GPU: {torch.cuda.get_device_name(0)}  |  CUDA {torch.version.cuda}")
        else:
            self.log("GPU не обнаружен — используется CPU")

    # ── Лог ───────────────────────────────────────────────────────────────────

    def log(self, message):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {message}")
        self.log_text.moveCursor(QTextCursor.End)

    def clear_log(self):
        self.log_text.clear()

    # ── Диалоги выбора файлов ─────────────────────────────────────────────────

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл модели", "", "Model Files (*.pt *.onnx *.engine);;All Files (*)"
        )
        if path:
            self.model_path_edit.setText(path)
            self.update_model_info(path)

    def browse_data_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите data.yaml", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if path:
            self.data_yaml_edit.setText(path)

    def browse_dataset_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку датасета")
        if not folder:
            return
        yamls = glob.glob(os.path.join(folder, "*.yaml")) + \
                glob.glob(os.path.join(folder, "*.yml"))
        if not yamls:
            QMessageBox.warning(self, "Ошибка", "В папке не найдено *.yaml файлов")
            return
        self.data_yaml_edit.setText(yamls[0])
        self.log(f"Найден yaml: {yamls[0]}")

    def update_model_info(self, path):
        try:
            if path.endswith('.pt'):
                model = YOLO(path)
                self.model_info_label.setText(f"PyTorch модель  |  task: {model.task}")
            elif path.endswith('.onnx'):
                if ORT_AVAILABLE:
                    ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                self.model_info_label.setText("ONNX модель (валидна)")
            elif path.endswith('.engine'):
                self.model_info_label.setText("TensorRT engine")
            else:
                self.model_info_label.setText("Неизвестный формат")
        except Exception as e:
            self.model_info_label.setText(f"Ошибка: {e}")

    # ── Ручной экспорт ────────────────────────────────────────────────────────

    def export_model(self):
        model_path = self.model_path_edit.text().strip()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Ошибка", "Выберите существующий файл модели")
            return

        imgsz    = _parse_int(self.imgsz_edit, 640)
        opset    = _parse_int(self.opset_edit, 17)
        dynamic  = self.dynamic_check.isChecked()
        simplify = self.simplify_check.isChecked()
        data_yaml = self.data_yaml_edit.text().strip()

        thread = ExportThread(
            model_path, 'ONNX FP32', FORMAT_CONFIGS['ONNX FP32'],
            imgsz, opset, dynamic, simplify, data_yaml
        )
        if not self._safe_replace_thread('export_thread', thread):
            return

        self.export_btn.setEnabled(False)
        self.export_progress.setVisible(True)
        self.export_progress.setValue(0)

        self.export_thread.log_signal.connect(self.log)
        self.export_thread.progress_signal.connect(self.export_progress.setValue)
        self.export_thread.finished_signal.connect(self.on_export_finished)
        self.export_thread.start()

    def on_export_finished(self, exported_path):
        self.export_btn.setEnabled(True)
        self.export_progress.setVisible(False)  # Bug 4 fix: всегда скрываем
        self.export_progress.setValue(0)
        if exported_path:
            self.exported_onnx_path = exported_path
            self.log(f"ONNX готов: {exported_path}")
            reply = QMessageBox.question(
                self, "Экспорт завершён",
                "ONNX модель сохранена.\nИспользовать её для бенчмаркинга?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.model_path_edit.setText(exported_path)
                self.update_model_info(exported_path)

    # ── Ручной бенчмарк ───────────────────────────────────────────────────────

    def run_benchmark(self, model_type):
        # Bug 5 fix: проверяем isRunning перед стартом
        if self.benchmark_thread and self.benchmark_thread.isRunning():
            QMessageBox.warning(self, "Занято", "Бенчмарк уже выполняется")
            return

        path = self.model_path_edit.text().strip()
        if model_type == "pt":
            if not path or not path.endswith('.pt'):
                QMessageBox.warning(self, "Ошибка", "Выберите .pt файл модели")
                return
            label = "PyTorch"
        else:
            if not path:
                QMessageBox.warning(self, "Ошибка", "Выберите файл модели")
                return
            if not path.endswith('.onnx'):
                if self.exported_onnx_path and os.path.exists(self.exported_onnx_path):
                    reply = QMessageBox.question(
                        self, "Использовать ONNX",
                        f"Использовать экспортированный ONNX?\n{self.exported_onnx_path}",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        path = self.exported_onnx_path
                    else:
                        return
                else:
                    QMessageBox.warning(self, "Ошибка", "Сначала экспортируйте модель в ONNX")
                    return
            label = "ONNX FP32"

        data_yaml = self.data_yaml_edit.text().strip()
        if not data_yaml or not os.path.exists(data_yaml):
            QMessageBox.warning(self, "Ошибка", "Укажите корректный путь к data.yaml")
            return

        device = self.device_combo.currentText()
        batch  = _parse_int(self.batch_edit, 16)
        imgsz  = _parse_int(self.imgsz_edit, 640)
        half   = self.half_pt_check.isChecked() if model_type == "pt" else False

        thread = BenchmarkThread(path, data_yaml, device, imgsz, half, batch, label)
        self._safe_replace_thread('benchmark_thread', thread)

        self.run_benchmark_pt_btn.setEnabled(False)
        self.run_benchmark_onnx_btn.setEnabled(False)

        self.benchmark_thread.log_signal.connect(self.log)
        self.benchmark_thread.finished_signal.connect(self._on_manual_benchmark_finished)
        self.benchmark_thread.start()

    def _on_manual_benchmark_finished(self, results):
        self.run_benchmark_pt_btn.setEnabled(True)
        self.run_benchmark_onnx_btn.setEnabled(True)
        if results:
            self.results_store[results['label']] = results
            self._refresh_results_table()
            self.log(f"mAP50={results['map50']:.4f}  mAP50-95={results['map50_95']:.4f}  "
                     f"time={results['val_time']:.1f}s")

    # ── Запуск всего ──────────────────────────────────────────────────────────

    def run_all(self):
        if self.run_all_thread and self.run_all_thread.isRunning():
            QMessageBox.warning(self, "Занято", "Запуск уже выполняется")
            return

        pt_path = self.model_path_edit.text().strip()
        if not pt_path or not pt_path.endswith('.pt') or not os.path.exists(pt_path):
            QMessageBox.warning(self, "Ошибка", "Выберите существующий .pt файл модели")
            return

        data_yaml = self.data_yaml_edit.text().strip()
        if not data_yaml or not os.path.exists(data_yaml):
            QMessageBox.warning(self, "Ошибка", "Укажите корректный путь к data.yaml")
            return

        formats_to_run = [
            label for label, cb in self.format_checkboxes.items() if cb.isChecked()
        ]
        if not formats_to_run:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы один формат")
            return

        self.results_store = {}
        self.results_table.setRowCount(0)

        thread = RunAllThread(
            pt_path=pt_path,
            formats_to_run=formats_to_run,
            data_yaml=data_yaml,
            device=self.device_combo.currentText(),
            imgsz=_parse_int(self.imgsz_edit, 640),
            batch=_parse_int(self.batch_edit, 16),
            opset=_parse_int(self.opset_edit, 17),
            dynamic=self.dynamic_check.isChecked(),
            simplify=self.simplify_check.isChecked(),
            half_pt=self.half_pt_check.isChecked(),
        )
        self._safe_replace_thread('run_all_thread', thread)

        self.run_all_thread.log_signal.connect(self.log)
        self.run_all_thread.progress_signal.connect(self.overall_progress.setValue)
        self.run_all_thread.step_signal.connect(self.step_label.setText)
        self.run_all_thread.result_signal.connect(self.on_format_result)
        self.run_all_thread.error_signal.connect(lambda msg: self.log(f"ВНИМАНИЕ: {msg}"))
        self.run_all_thread.finished_signal.connect(self.on_run_all_finished)

        self._set_run_buttons_enabled(False)
        self.overall_progress.setVisible(True)
        self.overall_progress.setValue(0)
        self.run_all_thread.start()

    def on_format_result(self, label, results):
        self.results_store[label] = results
        self._refresh_results_table()

    def on_run_all_finished(self):
        self._set_run_buttons_enabled(True)
        self.overall_progress.setVisible(False)

        # Вставить строки "ОШИБКА" для форматов, которые не дали результата
        checked = [l for l, cb in self.format_checkboxes.items() if cb.isChecked()]
        for label in checked:
            if label not in self.results_store:
                self._insert_error_row(label)

        self.log("--- Все операции завершены ---")

        # Автосохранение
        if self.results_store:
            saved = self._autosave_results()
            if saved:
                self.log(f"Результаты сохранены: {saved}")

    def _refresh_results_table(self):
        pt = self.results_store.get('PyTorch')

        # Строки в порядке FORMAT_CONFIGS
        ordered = [(l, self.results_store[l])
                   for l in FORMAT_CONFIGS if l in self.results_store]

        self.results_table.setRowCount(len(ordered))
        for row, (label, res) in enumerate(ordered):
            size_mb = res.get('file_size_mb')
            cells = [
                label,
                f"{res['map50']:.4f}",
                f"{res['map50_95']:.4f}",
                f"{res['val_time']:.1f}",
            ]

            color = None
            if pt and pt.get('map50') and res.get('map50') and label != 'PyTorch':
                delta = res['map50'] - pt['map50']
                speed = (pt['val_time'] / res['val_time']) if res['val_time'] > 0 else 0
                cells += [f"{delta:+.4f}", f"×{speed:.2f}"]
                color = COLOR_GREEN if delta >= 0 else (COLOR_RED if delta < -0.005 else None)
            else:
                cells += ["—", "—"]

            # Размер файла
            if size_mb is not None:
                cells.append(f"{size_mb:.1f}")
                if pt and pt.get('file_size_mb') and label != 'PyTorch':
                    ratio = size_mb / pt['file_size_mb']
                    cells.append(f"×{ratio:.2f}")
                else:
                    cells.append("—")
            else:
                cells += ["—", "—"]

            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if color:
                    item.setBackground(color)
                self.results_table.setItem(row, col, item)

    def _insert_error_row(self, label):
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        cells = [label, "ОШИБКА"] + [""] * (len(TABLE_HEADERS) - 2)
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            item.setBackground(COLOR_RED)
            self.results_table.setItem(row, col, item)

    def _set_run_buttons_enabled(self, enabled):
        self.run_all_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)
        self.run_benchmark_pt_btn.setEnabled(enabled)
        self.run_benchmark_onnx_btn.setEnabled(enabled)

    # ── Сохранение результатов ────────────────────────────────────────────────

    def _build_results_payload(self):
        """Формирует словарь со всеми результатами для сериализации."""
        pt = self.results_store.get('PyTorch')
        rows = []
        for label in FORMAT_CONFIGS:
            if label not in self.results_store:
                continue
            res = self.results_store[label]
            size_mb = res.get('file_size_mb')
            row = {
                'format':       label,
                'map50':        res.get('map50'),
                'map50_95':     res.get('map50_95'),
                'fitness':      res.get('fitness'),
                'val_time':     res.get('val_time'),
                'file_size_mb': round(size_mb, 2) if size_mb else None,
            }
            if pt and label != 'PyTorch' and pt.get('map50') and res.get('map50'):
                row['delta_map50']  = res['map50'] - pt['map50']
                row['speed_vs_pt']  = (pt['val_time'] / res['val_time']
                                       if res['val_time'] > 0 else None)
                pt_size = pt.get('file_size_mb')
                row['size_vs_pt'] = (round(size_mb / pt_size, 3)
                                     if size_mb and pt_size else None)
            else:
                row['delta_map50'] = None
                row['speed_vs_pt'] = None
                row['size_vs_pt']  = None
            rows.append(row)

        return {
            'timestamp':   datetime.now().isoformat(timespec='seconds'),
            'model':       self.model_path_edit.text().strip(),
            'dataset':     self.data_yaml_edit.text().strip(),
            'device':      self.device_combo.currentText(),
            'imgsz':       _parse_int(self.imgsz_edit, 640),
            'batch':       _parse_int(self.batch_edit, 16),
            'results':     rows,
        }

    def _results_dir(self):
        """Возвращает папку results/ рядом со скриптом, создаёт если нет."""
        d = Path(__file__).parent / 'results'
        d.mkdir(exist_ok=True)
        return d

    def _autosave_results(self):
        """Автоматически сохраняет JSON + CSV в results/. Возвращает базовое имя файла."""
        try:
            ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
            base = self._results_dir() / f'benchmark_{ts}'
            payload = self._build_results_payload()
            self._write_json(payload, base.with_suffix('.json'))
            self._write_csv(payload, base.with_suffix('.csv'))
            short = f"results/benchmark_{ts}.json/.csv"
            self.autosave_label.setText(f"Автосохранено: benchmark_{ts}")
            return short
        except Exception as e:
            self.log(f"Ошибка автосохранения: {e}")
            return None

    def _write_json(self, payload, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _write_csv(self, payload, path):
        rows = payload['results']
        if not rows:
            return
        fieldnames = ['format', 'map50', 'map50_95', 'fitness',
                      'val_time', 'file_size_mb', 'delta_map50', 'speed_vs_pt', 'size_vs_pt']
        meta = {k: payload[k] for k in ('timestamp', 'model', 'dataset', 'device', 'imgsz', 'batch')}

        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            # Мета-заголовок
            for k, v in meta.items():
                writer.writerow([f'# {k}', v])
            writer.writerow([])
            # Данные
            writer.writerow(fieldnames)
            for row in rows:
                writer.writerow([row.get(k, '') for k in fieldnames])

    def _save_results_json_dialog(self):
        if not self.results_store:
            QMessageBox.information(self, "Нет данных", "Сначала запустите бенчмарк")
            return
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        default = str(self._results_dir() / f'benchmark_{ts}.json')
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты (JSON)", default, "JSON (*.json)"
        )
        if not path:
            return
        try:
            self._write_json(self._build_results_payload(), path)
            self.log(f"JSON сохранён: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")

    def _save_results_csv_dialog(self):
        if not self.results_store:
            QMessageBox.information(self, "Нет данных", "Сначала запустите бенчмарк")
            return
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        default = str(self._results_dir() / f'benchmark_{ts}.csv')
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты (CSV)", default, "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self._write_csv(self._build_results_payload(), path)
            self.log(f"CSV сохранён: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")

    # ── Thread helpers ────────────────────────────────────────────────────────

    def _safe_replace_thread(self, attr, new_thread):
        """Отключает сигналы старого потока и заменяет ссылку. Возвращает False если поток занят."""
        old = getattr(self, attr, None)
        if old and old.isRunning():
            QMessageBox.warning(self, "Занято", "Операция уже выполняется")
            return False
        if old:
            for sig_name in ('log_signal', 'finished_signal', 'progress_signal',
                             'step_signal', 'result_signal', 'error_signal'):
                if hasattr(old, sig_name):
                    try:
                        getattr(old, sig_name).disconnect()
                    except TypeError:
                        pass
        setattr(self, attr, new_thread)
        return True

    def _stop_thread(self, thread):
        if thread and thread.isRunning():
            thread.requestInterruption()
            if not thread.wait(3000):
                thread.terminate()
                thread.wait()

    def closeEvent(self, event):
        self._stop_thread(self.export_thread)
        self._stop_thread(self.benchmark_thread)
        self._stop_thread(self.run_all_thread)
        event.accept()


# ─── Точка входа ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    window = YOLOBenchmarkApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
