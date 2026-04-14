import sys
import logging
import threading
from datetime import datetime
from pathlib import Path


def _setup_logging() -> logging.Logger:
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"

    fmt = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Подавляем шумные логи Ultralytics/PIL и отключаем propagation
    for noisy in ('ultralytics', 'PIL', 'urllib3', 'filelock'):
        lg = logging.getLogger(noisy)
        lg.setLevel(logging.WARNING)
        lg.propagate = False  # не дублируем в root-хендлеры

    return logging.getLogger('yolo_bench')


logger = _setup_logging()


def _excepthook(exc_type, exc_value, exc_tb):
    """Перехватывает неперехваченные исключения в главном потоке."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logger.critical("Необработанное исключение:", exc_info=(exc_type, exc_value, exc_tb))
    logging.shutdown()


def _thread_excepthook(args):
    """Перехватывает неперехваченные исключения в дочерних потоках (Python 3.8+)."""
    if args.exc_type is SystemExit:
        return
    logger.critical(
        "Необработанное исключение в потоке %s:",
        args.thread.name if args.thread else "неизвестный",
        exc_info=(args.exc_type, args.exc_value, args.exc_tb),
    )


sys.excepthook = _excepthook
threading.excepthook = _thread_excepthook
