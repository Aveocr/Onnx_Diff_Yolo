import os
import glob
import json
import csv
import logging
from datetime import datetime
from pathlib import Path

import torch
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar,
    QGroupBox, QComboBox, QLineEdit, QCheckBox, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTextCursor

from logging_setup import logger
from constants import (
    CUDA_AVAILABLE, FORMAT_CONFIGS, TABLE_HEADERS,
    COLOR_GREEN, COLOR_RED
)
from utils import _parse_int, _yaml_exists_or_builtin
from workers import (
    DepsInstallThread, BenchmarkThread, ExportThread, RunAllThread
)


class YOLOBenchmarkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Export & Benchmark Tool")
        self.setGeometry(100, 100, 980, 780)

        self.export_thread    = None
        self.benchmark_thread = None
        self.run_all_thread   = None
        self.deps_thread      = None

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
        self.simplify_check.setChecked(False)  # onnxslim медленный — включать вручную при необходимости
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

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Очистить лог")
        clear_btn.clicked.connect(self.clear_log)
        open_log_btn = QPushButton("Открыть лог-файл")
        open_log_btn.clicked.connect(self._open_log_file)
        self.install_deps_btn = QPushButton("Установить зависимости")
        self.install_deps_btn.setToolTip(
            "Устанавливает onnx, onnxruntime (CPU или GPU) и tensorrt через pip"
        )
        self.install_deps_btn.clicked.connect(self.install_deps)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(open_log_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.install_deps_btn)
        layout.addLayout(btn_row)

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
        logger.info(message)

    def clear_log(self):
        self.log_text.clear()

    def _open_log_file(self):
        log_dir = Path(__file__).parent / 'logs'
        log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        if log_file.exists():
            os.startfile(str(log_file))
        else:
            QMessageBox.information(self, "Лог", f"Лог-файл не найден:\n{log_file}")

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
        # Не загружаем модель в главном потоке — это блокирует Qt и может крашнуть
        # на Windows из-за multiprocessing spawn. Показываем только метаданные файла.
        try:
            size_mb = os.path.getsize(path) / 1024 / 1024
            ext = Path(path).suffix.lower()
            labels = {'.pt': 'PyTorch', '.onnx': 'ONNX', '.engine': 'TensorRT engine'}
            fmt = labels.get(ext, 'Неизвестный формат')
            self.model_info_label.setText(f"{fmt}  |  {size_mb:.1f} МБ")
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
        self.export_progress.setVisible(False)
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
        if not data_yaml or not _yaml_exists_or_builtin(data_yaml):
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
        if not data_yaml or not _yaml_exists_or_builtin(data_yaml):
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

    # ── Установка зависимостей ────────────────────────────────────────────────

    def install_deps(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Установка зависимостей")
        msg.setText(
            "<b>Выберите вариант установки:</b><br><br>"
            "<b>CPU</b> — onnx, onnxruntime, onnxslim<br>"
            "<b>GPU</b> — onnx, onnxruntime-gpu, onnxslim, tensorrt<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(требуется CUDA и NVIDIA GPU)<br><br>"
            "После установки нужно <b>перезапустить</b> приложение."
        )
        msg.setTextFormat(Qt.RichText)
        cpu_btn = msg.addButton("CPU", QMessageBox.AcceptRole)
        gpu_btn = msg.addButton("GPU (CUDA)", QMessageBox.AcceptRole)
        msg.addButton("Отмена", QMessageBox.RejectRole)
        msg.exec_()

        clicked = msg.clickedButton()
        if clicked == cpu_btn:
            packages = DepsInstallThread.PACKAGES_CPU
        elif clicked == gpu_btn:
            packages = DepsInstallThread.PACKAGES_GPU
        else:
            return

        thread = DepsInstallThread(packages)
        if not self._safe_replace_thread('deps_thread', thread):
            return

        self.install_deps_btn.setEnabled(False)
        self.deps_thread.log_signal.connect(self.log)
        self.deps_thread.finished_signal.connect(self._on_deps_finished)
        self.deps_thread.start()

    def _on_deps_finished(self):
        self.install_deps_btn.setEnabled(True)
        QMessageBox.information(
            self, "Готово",
            "Зависимости установлены.\nПерезапустите приложение, чтобы изменения вступили в силу."
        )

    def closeEvent(self, event):
        self._stop_thread(self.export_thread)
        self._stop_thread(self.benchmark_thread)
        self._stop_thread(self.run_all_thread)
        self._stop_thread(self.deps_thread)
        logging.shutdown()
        event.accept()
