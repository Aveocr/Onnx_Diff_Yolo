"""
Подготовка тестового датасета для YOLO-бенчмарка.

Режим 1 (по умолчанию): скачивает COCO128 — официальный мини-датасет Ultralytics
    128 изображений, 80 классов, ~26 MB
    Результат: datasets/coco128/  +  coco128.yaml рядом со скриптом

Режим 2 (--synthetic): генерирует синтетический датасет без интернета
    50 изображений, 3 класса (box / circle / triangle)
    Результат: datasets/synthetic/  +  synthetic.yaml рядом со скриптом

Использование:
    python prepare_dataset.py           # COCO128
    python prepare_dataset.py --synthetic  # синтетика
"""

import sys
import os
import random
import argparse
import shutil
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────


def prepare_coco128(base_dir: Path):
    """Скачивает COCO128 через Ultralytics."""
    print("Скачиваем COCO128 через Ultralytics...")
    try:
        from ultralytics.utils.downloads import download
        import yaml

        # Ultralytics хранит coco128.yaml внутри пакета
        import ultralytics
        builtin_yaml = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco128.yaml"
        if not builtin_yaml.exists():
            # Попробуем найти в другом месте
            import glob as _glob
            found = _glob.glob(str(Path(ultralytics.__file__).parent / "**" / "coco128.yaml"), recursive=True)
            builtin_yaml = Path(found[0]) if found else None

        datasets_dir = base_dir / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)

        # Запускаем val на крошечной модели — Ultralytics автоматически скачает датасет
        from ultralytics import YOLO
        print("  Загружаем YOLOv8n и запускаем одну итерацию val для скачивания датасета...")
        model = YOLO("yolov8n.pt")
        model.val(
            data="coco128.yaml",
            imgsz=640,
            batch=4,
            device="cpu",
            verbose=False,
            plots=False,
        )

        # После val датасет лежит в ~/ultralytics/datasets/ или рядом
        # Ищем скачанный датасет
        possible_roots = [
            Path.home() / "ultralytics" / "datasets" / "coco128",
            Path.home() / "datasets" / "coco128",
            Path("datasets") / "coco128",
            Path("../datasets/coco128"),
        ]
        coco128_root = None
        for p in possible_roots:
            if p.exists():
                coco128_root = p
                break

        # Генерируем свой data.yaml с абсолютным путём
        out_yaml = base_dir / "coco128.yaml"

        if coco128_root:
            # Читаем оригинальный yaml и патчим путь
            if builtin_yaml and builtin_yaml.exists():
                with open(builtin_yaml) as f:
                    data = yaml.safe_load(f)
                data["path"] = str(coco128_root.parent)
            else:
                with open(builtin_yaml or "coco128.yaml") as f:
                    data = yaml.safe_load(f)
                data["path"] = str(coco128_root.parent)

            with open(out_yaml, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
            print(f"\nCOCO128 ready: {coco128_root}")
        else:
            # Датасет скачан, но путь неизвестен — просто копируем yaml
            if builtin_yaml and builtin_yaml.exists():
                shutil.copy(builtin_yaml, out_yaml)
            print(f"\nDataset downloaded, using Ultralytics built-in path")

        print(f"data.yaml: {out_yaml}")
        return out_yaml

    except Exception as e:
        print(f"\n✗ Не удалось скачать COCO128: {e}")
        print("  Запустите с флагом --synthetic для синтетического датасета")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────


def prepare_synthetic(base_dir: Path, n_images: int = 50, imgsz: int = 640):
    """Генерирует синтетический датасет: цветные фигуры на случайном фоне."""
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        import yaml
    except ImportError as e:
        print(f"✗ Не хватает зависимости: {e}")
        sys.exit(1)

    CLASSES = ["box", "circle", "triangle"]
    N_CLASSES = len(CLASSES)

    dataset_dir = base_dir / "datasets" / "synthetic"
    imgs_val    = dataset_dir / "images" / "val"
    lbls_val    = dataset_dir / "labels" / "val"
    imgs_train  = dataset_dir / "images" / "train"
    lbls_train  = dataset_dir / "labels" / "train"

    for d in (imgs_val, lbls_val, imgs_train, lbls_train):
        d.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    def random_color():
        return tuple(random.randint(50, 255) for _ in range(3))

    def draw_box(draw, cx, cy, w, h):
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        draw.rectangle([x1, y1, x2, y2], fill=random_color(), outline=(0, 0, 0), width=2)

    def draw_circle(draw, cx, cy, w, h):
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        draw.ellipse([x1, y1, x2, y2], fill=random_color(), outline=(0, 0, 0), width=2)

    def draw_triangle(draw, cx, cy, w, h):
        pts = [
            (cx, cy - h // 2),
            (cx - w // 2, cy + h // 2),
            (cx + w // 2, cy + h // 2),
        ]
        draw.polygon(pts, fill=random_color(), outline=(0, 0, 0))

    DRAWERS = [draw_box, draw_circle, draw_triangle]

    def generate_image(idx, split):
        img_dir = imgs_val if split == "val" else imgs_train
        lbl_dir = lbls_val if split == "val" else lbls_train

        bg = tuple(random.randint(160, 240) for _ in range(3))
        img = Image.new("RGB", (imgsz, imgsz), bg)
        draw = ImageDraw.Draw(img)

        # добавляем шум
        noise = np.random.randint(0, 30, (imgsz, imgsz, 3), dtype=np.uint8)
        noise_img = Image.fromarray(noise)
        img = Image.blend(img.convert("RGBA"), noise_img.convert("RGBA"), 0.05).convert("RGB")
        draw = ImageDraw.Draw(img)

        n_objects = random.randint(1, 4)
        labels = []

        for _ in range(n_objects):
            cls_id = random.randint(0, N_CLASSES - 1)
            w = random.randint(imgsz // 10, imgsz // 3)
            h = random.randint(imgsz // 10, imgsz // 3)
            cx = random.randint(w // 2 + 5, imgsz - w // 2 - 5)
            cy = random.randint(h // 2 + 5, imgsz - h // 2 - 5)

            DRAWERS[cls_id](draw, cx, cy, w, h)

            # YOLO-формат: class cx cy w h (нормализованные 0-1)
            labels.append(
                f"{cls_id} "
                f"{cx / imgsz:.6f} {cy / imgsz:.6f} "
                f"{w / imgsz:.6f} {h / imgsz:.6f}"
            )

        name = f"img_{idx:04d}"
        img.save(img_dir / f"{name}.jpg", quality=90)
        with open(lbl_dir / f"{name}.txt", "w") as f:
            f.write("\n".join(labels))

    print(f"Генерируем синтетический датасет ({n_images} изображений)...")
    n_val   = max(10, n_images // 5)
    n_train = n_images - n_val

    for i in range(n_train):
        generate_image(i, "train")
        if (i + 1) % 10 == 0:
            print(f"  train: {i + 1}/{n_train}", end="\r")
    print(f"  train: {n_train}/{n_train} OK")

    for i in range(n_val):
        generate_image(n_train + i, "val")
        if (i + 1) % 10 == 0:
            print(f"  val:   {i + 1}/{n_val}", end="\r")
    print(f"  val:   {n_val}/{n_val}   OK")

    # data.yaml
    data = {
        "path":  str(dataset_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    N_CLASSES,
        "names": CLASSES,
    }
    out_yaml = base_dir / "synthetic.yaml"
    import yaml
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"\nDataset: {dataset_dir}")
    print(f"data.yaml: {out_yaml}")
    print(f"  Classes: {', '.join(CLASSES)}")
    print(f"  train: {n_train} images")
    print(f"  val:   {n_val} images")
    return out_yaml


# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Подготовка тестового датасета для YOLO-бенчмарка")
    parser.add_argument("--synthetic", action="store_true",
                        help="Сгенерировать синтетический датасет вместо скачивания COCO128")
    parser.add_argument("--n", type=int, default=50,
                        help="Количество изображений для синтетического датасета (по умолчанию: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Размер изображений (по умолчанию: 640)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    if args.synthetic:
        yaml_path = prepare_synthetic(base_dir, n_images=args.n, imgsz=args.imgsz)
    else:
        yaml_path = prepare_coco128(base_dir)

    print(f"\nDone! Use in the app:\n  data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
