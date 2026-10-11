# Лабораторные работы: математические методы в CV (CSE5382)

## Установка

Нужен Python 3.12 и [uv](https://docs.astral.sh/uv/):

```
uv sync
```

Без uv:

```
python -m venv .venv
.venv/bin/pip install ipykernel matplotlib numpy opencv-python torch torchvision
```

(в Windows вместо `.venv/bin/pip` — `.venv\Scripts\pip`)

## Запуск

Ноутбуки `lab01`–`lab15` открываются в Jupyter или VS Code. В качестве ядра выберите
окружение `.venv` из этой папки и запускайте ноутбуки из корня репозитория.

Работы 11 и 12 при первом запуске скачивают датасеты.
