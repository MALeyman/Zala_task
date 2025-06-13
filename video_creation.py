
import cv2, os
import numpy as np


def creation_video(image_path, num_vid=0, win_size=1200, num_frames = 700, step = 5, fps = 30):
    ''' 
        Создание видео из изображения.
        win_size: размер видео кадра
        num_frames: количество кадров
        step: шаг движения по изображению
        fps: частота кадров
    '''
    image = cv2.imread(image_path) 
    (h, w) = image.shape[:2]

    # Размер окна (области просмотра)
    win_size = 1200  

    # Параметры для движения окна
    step = 5  # Шаг движения 
    fps = 30  # Частота кадров
    num_frames = 700  # Количество кадров

    # Создание видео
    output_video_path = "videos/output5.avi"
    os.makedirs("videos", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (win_size, win_size))
    print(output_video_path)
    # Начальные координаты
    x, y = 0, 0
    # начало движения
    if num_vid==0:
        direction = "right"
    else:
        direction = "diagonal"

    print("direction", direction)
    print("image_path", image_path)
    # Генерация кадров
    for _ in range(num_frames):
        # Вырезаем окно из изображения
        cropped = image[y:y + win_size, x:x + win_size]
        
        #   окно корректного размера?
        if cropped.shape[:2] == (win_size, win_size):
            out.write(cropped)

        if num_vid == 0:
            # Логика движения
            if direction == "right":
                x += step
                if x + win_size >= w:
                    x = w - win_size
                    direction = "down"
            elif direction == "down":
                y += step
                if y + win_size >= h:
                    y = h - win_size
                    direction = "left"
            elif direction == "left":
                x -= step
                if x <= 0:
                    x = 0
                    direction = "up"
            elif direction == "up":
                y -= step
                if y <= 0:
                    y = 0
                    direction = "right"
        else:
            # Логика движения по диагонали и кругу
            if direction == "diagonal":
                x += step
                y += step
                if x + win_size >= w or y + win_size >= h:
                    x = min(x, w - win_size)
                    y = min(y, h - win_size)
                    direction = "left"
            elif direction == "left":
                x -= step
                if x <= 0:
                    x = 0
                    direction = "up"
            elif direction == "up":
                y -= step
                if y <= 0:
                    y = 0
                    direction = "right"
            elif direction == "right":
                x += step
                if x + win_size >= w:
                    x = w - win_size
                    direction = "down"
            elif direction == "down":
                y += step
                if y + win_size >= h:
                    y = h - win_size
                    direction = "diagonal" 
                
    # Закрываем видеофайл
    out.release()

    print("Видео успешно создано!")