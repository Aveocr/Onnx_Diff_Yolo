import sys
import multiprocessing

try:
    from PyQt5.QtWidgets import QApplication
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Установите зависимости: pip install PyQt5 ultralytics torch")
    sys.exit(1)

import logging_setup  # noqa: F401 — регистрирует логгер и хуки исключений
from app import YOLOBenchmarkApp


def main():
    # На Windows multiprocessing использует spawn — freeze_support() обязателен,
    # иначе рабочие процессы DataLoader переимпортируют модуль и крашнутся
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    window = YOLOBenchmarkApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
