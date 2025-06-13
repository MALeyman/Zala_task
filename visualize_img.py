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















