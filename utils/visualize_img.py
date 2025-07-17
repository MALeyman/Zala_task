import os
import cv2
import matplotlib.pyplot as plt
import time
import IPython.display as display
from PIL import Image



# Функция для загрузки аннотаций
def load_annotations(label_file):
    """ 
        Функция для загрузки аннотаций
    """
    with open(label_file, "r") as file:
        lines = file.readlines()
    
    bboxes = []
    for line in lines:
        data = line.strip().split()
        if len(data) != 5:
            continue
        class_id = int(data[0])  # Класс объекта
        x_center, y_center, width, height = map(float, data[1:])
        bboxes.append((class_id, x_center, y_center, width, height))
    return bboxes


# Функция для отображения изображения с боксами
def show_image_with_boxes(image_file, label_file, size_x=10, size_y=10):
    """ 
        Функция для отображения изображения с боксами
    """
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    
    h, w, _ = image.shape                       # размеры изображения
    
    # аннотации
    bboxes = load_annotations(label_file)
    
    # Рисуем боксы
    for class_id, x_center, y_center, width, height in bboxes:
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        color = (255, 0, 0)  # цвет боксов
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, str(class_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, thickness)

    plt.figure(figsize=(size_x, size_y))
    plt.imshow(image)
    plt.axis("off")
    plt.show()


# Просмотр изображения с боксами
def visualize_img(image_name, images_path, labels_path, size_x=10, size_y=10):
    """ 
        Просмотр изображения с боксами
    """
    # пути к файлам
    image_file = os.path.join(images_path, image_name)
    label_file = os.path.join(labels_path, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))

    print(image_file)
    print(label_file)
    # Проверяем, существует ли файл и аннотация
    if os.path.exists(image_file) and os.path.exists(label_file):
        show_image_with_boxes(image_file, label_file, size_x = size_x, size_y=size_y)
    else:
        print("Файл изображения или аннотации не найден!")


# Просмотр всех изображений с боксами
def visualize_img_full(images_path="dataset/datasets_full/images/train", labels_path="dataset/datasets_full/labels/train", size_x = 10, size_y=10):
    """  
        Просмотр всех изображений с боксами
    """
    # Получаем список файлов изображений
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))])
    
    if image_files:
        for image_name in image_files:
            display.clear_output(wait=True)  # Очистка вывода
            try:
                print(image_name)
                visualize_img(image_name, images_path, labels_path, size_x = size_x, size_y=size_y)                
                time.sleep(4)  # Задержка 4 секунды                
            except FileNotFoundError:
                print(f"Файл {image_name} не найден, пропускаем...")
            # break
    print("Все изображения проверены!")







import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def visualize_sample(image, target, threshold=0.5, class_names=None, num_anchors=3, num_classes=8):
    """
    Визуализация предсказаний или разметки heatmap'а формата [B*C, S, S].
    
    image: Tensor [3, H, W]
    target: Tensor [B*C, S, S]
    """
    image = image.permute(1, 2, 0).cpu().numpy()
    H, W = image.shape[:2]
    
    BxC, S, _ = target.shape
    assert BxC == num_anchors * num_classes, "target shape должен быть [B*C, S, S]"

    # Восстановим форму [B, C, S, S]
    target = target.view(num_anchors, num_classes, S, S)

    grid_y = np.linspace(0, H, S + 1, dtype=int)
    grid_x = np.linspace(0, W, S + 1, dtype=int)

    class_cells = []
    for gy in range(S):
        for gx in range(S):
            for a in range(num_anchors):
                for c in range(num_classes):
                    if target[a, c, gy, gx] > threshold:
                        x1 = grid_x[gx]
                        y1 = grid_y[gy]
                        x2 = grid_x[gx + 1]
                        y2 = grid_y[gy + 1]
                        class_cells.append((c, gy, gx, x1, y1, x2 - x1, y2 - y1))

    # 1) Первая фигура: изображение с сеткой и ячейками 
    fig_img, ax_img = plt.subplots(figsize=(12, 12))
    ax_img.imshow(image)
    ax_img.set_title('Изображение с ячейками')

    for x in grid_x:
        ax_img.axvline(x=x, color='white', linestyle='--', linewidth=0.2)
    for y in grid_y:
        ax_img.axhline(y=y, color='white', linestyle='--', linewidth=0.2)

    for c, gy, gx, x1, y1, w, h in class_cells:
        ax_img.add_patch(patches.Rectangle((x1, y1), w, h, edgecolor='lime', linewidth=1, fill=False))
        ax_img.text(x1 + 2, y1 + 12, f"{c if class_names is None else class_names[c]}", color='red', fontsize=7)

    plt.tight_layout()
    plt.show()


        # 2) Вторая фигура: список объектов и ячеек 
    fig_text, ax_text = plt.subplots(figsize=(6, max(3, len(class_cells) * 0.25)))
    ax_text.axis('off')
    ax_text.set_title('Обнаруженные классы и ячейки', fontsize=12, pad=10)

    for idx, (c, gy, gx, *_ ) in enumerate(class_cells):
        name = class_names[c] if class_names else f"Класс {c}"
        ax_text.text(0.01, 1 - idx * 0.05, f"{name} -> Ячейка (Y={gy}, X={gx})",
                     fontsize=10, verticalalignment='top', transform=ax_text.transAxes)


    plt.tight_layout()
    plt.show()







