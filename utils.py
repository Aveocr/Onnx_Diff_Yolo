import os
from pathlib import Path


def _parse_int(line_edit, default):
    try:
        return int(line_edit.text())
    except ValueError:
        return default


def _is_builtin_dataset(yaml_path: str) -> bool:
    """Возвращает True для встроенных датасетов Ultralytics (без разделителей пути).
    Например: 'coco128.yaml', 'coco128' — Ultralytics сам найдёт и скачает."""
    return os.sep not in yaml_path and '/' not in yaml_path


def _yaml_exists_or_builtin(yaml_path: str) -> bool:
    """Возвращает True если yaml существует на диске ИЛИ является встроенным датасетом."""
    return _is_builtin_dataset(yaml_path) or os.path.exists(yaml_path)


def _check_yaml_paths(yaml_path: str) -> str:
    """Проверяет yaml датасета. Ошибку возвращает только если:
    - yaml-файл не существует вообще (и не является встроенным датасетом)
    - path абсолютный и не существует на диске
    Встроенные датасеты Ultralytics ('coco128.yaml') и относительные пути не проверяем."""
    try:
        # Встроенные датасеты Ultralytics (coco128.yaml и т.п.) — пропускаем проверку
        if _is_builtin_dataset(yaml_path):
            return ""

        if not Path(yaml_path).exists():
            return f"Файл не найден: {yaml_path}"

        import yaml
        with open(yaml_path, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        base = data.get('path', '')
        p = Path(base)
        # Проверяем только абсолютные пути — относительные Ultralytics ищет сам
        if p.is_absolute() and not p.exists():
            return (
                f"Путь датасета не найден на этой машине:\n  {base}\n"
                "Запустите prepare_dataset.py или выберите другой data.yaml."
            )
        return ""
    except Exception as e:
        return f"Не удалось прочитать {yaml_path}: {e}"
