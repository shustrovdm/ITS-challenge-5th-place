##**Описание решения(5-е место в задаче ITS)**

####1. Классификация

Модель: se_resnext50_32x4d pretrained imagenet  
Оптимизатор: Adam  
Функция потерь: CrossEntropyLoss  
Learning Rate: 0.0003  
Количество эпох обучения: 50  

Максимальное значение метрики f1 score на валидационном сете после 50ти эпох: >0.995

Обученная модель доступна в _logs/full_se_resnext50_32x4d_stratified/checkpoints/best_full.pth_


####2. Семантическая сегментация

Модель: efficientnet-b7 encoder(imagenet pretrained) + Unet decoder  
Оптимизатор: Adam  
Функция потерь: 0.7* BinaryLovaszLoss + 0.3*BinaryFocalLoss  
Learning Rate: 0.0005  
Scheduler: Reduce On Plateu(patience = 15, factor = 0.5)  
Количество эпох обучения: 120  
TTA: hflip

Максимальное значение метрики mIoU score на валидационном сете после 120 эпох: >0.862

Обученная модель доступна в _logs_segmentation/best_model_efficientb7_100_200_unet.pth_



####**Обучение моделей:**

Перед выполнением поместить _train/_ в корень проекта.

1. Запустить _prepare_data.py_ для подготвки датасетов для сегментации/классификации
(Распределение изображений по директориям с названиями классов бактерий). 
Скрипт создаст директории _train_classes/_ и _train_classes_with_json/_ 
2. Запустить _train_classification.py_ для обучения классификационной модели.
Все пути в скрипте прописаны как относительные. Скрипт будет сохранять логи и чекпоинты модели
в папку _logs/_
3.  Запустить _train_segmentation.py_ для обучения классификационной модели.
Все пути в скрипте прописаны как относительные. Скрипт будет сохранять чекпоинты модели
в папку _logs_segmentation/_

####**Инференс моделей на тестовом сете test/ и запись предсказаний в submission.csv**

Поместить _test/_ в корень проекта.

1. Запустить _smpl_submission_masks.py_ для генерации масок из _sample_submission.csv_
2. Запустить _submission.py_ (По умолчанию относительные пути прописаны до моделей в _logs/full_se_resnext50_32x4d_stratified/checkpoints/best_full.pth_ 
и _logs_segmentation/best_model_efficientb7_60_250_unet.pth)_

