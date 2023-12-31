# Обнаружение дефектов на поверхности трубы

## Введение
Проект содержит скрипты для обучения / инференса моделей для хакатона. Цель данного исследования заключалась в разработке модели для автоматического обнаружения дефектов на поверхности трубы. Были предоставили только абстрактные точки дефектов, на основе которых использовалась модель SAM для определения областей дефектов на изображениях.

## Методика обучения
### Подготовка данных:
- Получены точки центров дефектов на поверхности трубы.
- Использовалась модель SAM для определения областей дефектов на основе этих точек.


### Обучение модели:
- Применена архитектура U-Net для обучения модели.
- Использовались различные бэкбоны: ResNet, EfficientNet.
- Экспериментировали с комбинациями функций потерь, такими как Dice Loss и Cross-Entropy Loss.

## Результаты
После проведения экспериментов с разными моделями и функциями потерь была выбрана модель U-Net с бэкбоном EfficientNet, как показавшая наилучшие результаты:
- **Модель:** U-Net с бэкбоном EfficientNet
- **Метрика:** Intersection over Union (IoU) составила 0.47.
