"""Готовит папку data/ для ноутбуков lab01–lab15.

Запуск из hw/cv/math_labs:  uv run python prepare_data.py

Источники:
- стандартные примеры OpenCV (samples/data): building, blox, box, box_in_scene,
  aloeL/aloeR, left01..left14, vtest.avi;
- hw/cv/week02/img.jpg — тёмный кадр для лабораторной 2;
- seg_scene.png и seg_mask.png — полусинтетическая сцена с точной маской
  для лабораторных 10 и 15 (собирается здесь же).
"""

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
CACHE = DATA / "_download"
BASE = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data"

# В наборе OpenCV нет left10.jpg.
CHESS = [f"left{i:02d}.jpg" for i in range(1, 15) if i != 10]
PLAIN = ["building.jpg", "blox.jpg", "box.png", "box_in_scene.png"]
EXTRA = ["aloeL.jpg", "aloeR.jpg", "fruits.jpg", "vtest.avi"]


def fetch(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if not path.exists():
        print("скачиваю", name)
        urlretrieve(f"{BASE}/{name}", path)
    return path


def make_segmentation_scene(seed: int = 0):
    """Апельсин (вырезка из fruits.jpg) в эллипсе на узорчатой ткани (aloeL.jpg)."""
    rng = np.random.default_rng(seed)
    cloth = cv2.imread(str(fetch("aloeL.jpg")))[40:520, 40:680]  # 480x640
    orange = cv2.imread(str(fetch("fruits.jpg")))[120:400, 110:300]
    h, w = cloth.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (330, 250), (150, 110), 20, 0, 360, 255, -1)
    cv2.rectangle(mask, (120, 90), (200, 150), 255, -1)

    texture = cv2.resize(orange, (w, h), interpolation=cv2.INTER_CUBIC)
    scene = np.where(mask[..., None] > 0, texture, cloth).astype(np.float32)

    # Неравномерное освещение слева направо и шум: с ними простой порог ошибается.
    ramp = np.linspace(0.65, 1.15, w, dtype=np.float32)[None, :, None]
    scene = scene * ramp + rng.normal(0, 6, scene.shape)
    return np.clip(scene, 0, 255).astype(np.uint8), mask


def main():
    DATA.mkdir(exist_ok=True)
    for name in PLAIN:
        cv2.imwrite(str(DATA / name), cv2.imread(str(fetch(name)), cv2.IMREAD_UNCHANGED))

    (DATA / "chessboard").mkdir(exist_ok=True)
    for name in CHESS:
        cv2.imwrite(str(DATA / "chessboard" / name), cv2.imread(str(fetch(name))))

    # Стереопара уменьшена вдвое: StereoSGBM на полном кадре считает слишком долго.
    for name in ("aloeL.jpg", "aloeR.jpg"):
        img = cv2.imread(str(fetch(name)))
        small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(DATA / name), small, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Три соседних кадра видео вместо самого ролика (8 МБ).
    cap = cv2.VideoCapture(str(fetch("vtest.avi")))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    for i in range(3):
        ok, frame = cap.read()
        assert ok
        cv2.imwrite(str(DATA / f"vtest_{i}.png"), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    dark_src = HERE.parent / "week02" / "img.jpg"
    dark = cv2.imdecode(np.fromfile(dark_src, np.uint8), cv2.IMREAD_COLOR)
    dark = cv2.resize(dark, (800, 800), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(DATA / "dark.jpg"), dark, [cv2.IMWRITE_JPEG_QUALITY, 92])

    scene, mask = make_segmentation_scene()
    cv2.imwrite(str(DATA / "seg_scene.png"), scene)
    cv2.imwrite(str(DATA / "seg_mask.png"), mask)

    for p in sorted(DATA.rglob("*")):
        if p.is_file() and "_download" not in p.parts and "datasets" not in p.parts:
            print(f"{p.relative_to(DATA)!s:28} {p.stat().st_size / 1024:8.0f} КБ")


if __name__ == "__main__":
    main()
