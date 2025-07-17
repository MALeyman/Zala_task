



import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torchvision.transforms as T








class MobileNetDataset(Dataset):
    """
    Кастомный PyTorch Dataset для загрузки изображений и аннотаций в формате YOLO.

    Параметры:
        images_path : str - Путь к директории с изображениями.
        
        labels_path : str - Путь к директории с аннотациями в формате YOLO (.txt файлы).
        
        S : int - Размер сетки ,  для деления изображения на ячейки.
        
        B : int - Количество предсказываемых боксов на ячейку.
        
        C : int - Количество классов объектов.
        
        transform : torchvision.transforms или None  - Аугментации и преобразования, применяемые к изображениям.
        
        image_size : int - Размер изображения (ширина и высота) после изменения размера. По умолчанию 512.
    
    Атрибуты:
        image_files : list - Список имён файлов изображений.
        
        resize : torchvision.transforms.Resize -  Преобразование для изменения размера изображений.
    """

    def __init__(self, images_path, labels_path, S=23, B=2, C=8, transform=None, image_size=736):
        self.images_path = images_path
        self.labels_path = labels_path
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
        self.image_files = sorted(os.listdir(images_path))

        # Преобразование для изменения размера
        self.resize = T.Resize((image_size, image_size))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_path, img_name)
        label_path = os.path.join(self.labels_path, img_name.replace('.jpg', '.txt'))

        # Загружаем и ресайзим изображение
        img = Image.open(img_path).convert('RGB')
        img = self.resize(img)
        img = torch.tensor(np.array(img) / 255.0, dtype=torch.float32).permute(2, 0, 1)

        # Загружаем аннотации
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # Пропускаем пустые или битые строки

                    try:
                        cls = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:])
                        if cls < 0 or cls >= self.C:
                            continue  # Пропускаем недопустимые классы
                        labels.append([cls, x_center, y_center, w, h])
                    except ValueError:
                        continue


        # Формируем target
        target = torch.zeros(self.S, self.S, self.B * (5 + self.C))
        for label in labels:
            cls, x_center, y_center, w, h = label

            # Номера ячеек, куда попадает центр бокса
            grid_x = int(x_center * self.S)
            grid_y = int(y_center * self.S)

            if grid_x >= self.S or grid_y >= self.S:
                continue  # За пределами

            for b in range(self.B):
                base = b * (5 + self.C)
                if target[grid_y, grid_x, base + 4] == 0:  # Только один объект на ячейку
                    # Смещение внутри ячейки
                    dx = x_center * self.S - grid_x
                    dy = y_center * self.S - grid_y

                    # Масштабируем ширину и высоту под сетку
                    w_cell = w 
                    h_cell = h 

                    target[grid_y, grid_x, base + 0] = dx
                    target[grid_y, grid_x, base + 1] = dy
                    target[grid_y, grid_x, base + 2] = w_cell
                    target[grid_y, grid_x, base + 3] = h_cell
                    target[grid_y, grid_x, base + 4] = 1
                    target[grid_y, grid_x, base + 5 + int(cls)] = 1
                    # break


        # target = target.unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        return img, target




def yolo_collate_fn(batch):
    """  
        Функция объединения  для DataLoader.

        Принимает батч данных,  кортеж (изображение, таргеты),
        и объединяет их в батч-тензоры для обучения модели.

    Возвращает:
        imgs (Tensor): Батч изображений с формой [batch_size, C, H, W].
        targets (Tensor): Батч разметок с формой [batch_size, S, S, B*(5 + C)].
    """
    imgs = [img for img, target in batch]
    targets = [target for img, target in batch]

    imgs = torch.stack(imgs, dim=0)
    targets = torch.stack(targets, dim=0)  # [batch_size, S, S, 12]
    return imgs, targets


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from torch.utils.data import DataLoader
from IPython.display import clear_output

def show_dataset(dataset, S=16, B=2, C=8, delay=2, batch_size=1):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=yolo_collate_fn)
    for imgs, targets in dataloader:
        for i in range(imgs.size(0)):
            img_tensor = imgs[i].cpu()
            true_tensor = targets[i].cpu()
            
            # print("БОКСЫ ", true_tensor[0])
            # print("min:", true_tensor.min().item())
            # print("max:", true_tensor.max().item())

            # Визуализация
            draw_gt_boxes(img_tensor, true_tensor, S=S, B=B, C=C)
            time.sleep(delay)
            clear_output(wait=True)
        break



import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_gt_boxes(img_tensor,
                  true_tensor,
                  S=16,
                  B=2,
                  C=8,
                  class_names=None,      # ← список имён классов или None
                  show_grid=False):      # ← если нужно показать сетку
    """
    Рисует ground-truth боксы и выводит номер/имя класса.
    """
    img = img_tensor.permute(1, 2, 0).numpy()
    h, w = img.shape[:2]

    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img)

    cell_w = w // S
    cell_h = h // S     # на случай прямоугольного изображения

    # опц. сетка
    if show_grid:
        for i in range(1, S):
            ax.axhline(i * cell_h, color='white', lw=0.5)
            ax.axvline(i * cell_w, color='white', lw=0.5)

    for gy in range(S):
        for gx in range(S):
            for b in range(B):
                base = b * (5 + C)

                # objectness (conf)
                conf = true_tensor[gy, gx, base + 4]
                if conf <= 0:           # 0 значит нет объекта
                    continue

                tx, ty, tw, th = true_tensor[gy, gx, base:base+4]
                # абсолютные координаты
                cx_abs = (gx + tx.item()) * cell_w
                cy_abs = (gy + ty.item()) * cell_h
                bw_abs = tw.item() * w
                bh_abs = th.item() * h

                x_min = cx_abs - bw_abs / 2
                y_min = cy_abs - bh_abs / 2

                rect = patches.Rectangle(
                    (x_min, y_min),
                    bw_abs, bh_abs,
                    linewidth=2,
                    edgecolor='lime',
                    facecolor='none')
                ax.add_patch(rect)

                # ------- класс -------
                class_vec = true_tensor[gy, gx, base + 5 : base + 5 + C]
                cls_idx   = int(class_vec.argmax())
                label_str = class_names[cls_idx] if class_names else str(cls_idx)

                ax.text(x_min, y_min - 2,
                        label_str,
                        color='lime',
                        fontsize=12,
                        verticalalignment='bottom')

    ax.set_title("Ground-Truth Boxes")
    ax.axis('off')
    plt.show()





class YOLOHeatmapDataset(Dataset):
    """
    Dataset для anchor-based multi-label классификации ячеек по YOLO-аннотациям.
    Каждой ячейке может быть присвоено до B классов (по якорям).

    Возвращает:
        img: Tensor [3, H, W]
        target: Tensor [B*C, S, S]
    """

    def __init__(self, images_dir, labels_dir, image_size=736, grid_size=30, num_classes=8, num_anchors=3, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        self.label_paths = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)

        target = torch.zeros((self.num_anchors, self.num_classes, self.grid_size, self.grid_size))

        with open(self.label_paths[idx], 'r') as f:
            for line in f:
                cls, cx, cy, w, h = map(float, line.strip().split())
                cls = int(cls)

                gx = int(cx * self.grid_size)
                gy = int(cy * self.grid_size)

                # добавляем метку в свободный anchor (или в последний)
                placed = False
                for anchor in range(self.num_anchors):
                    if target[anchor, cls, gy, gx] == 0:
                        target[anchor, cls, gy, gx] = 1
                        placed = True
                        break
                if not placed:
                    target[-1, cls, gy, gx] = 1

        # выход  [B*C, S, S]
        target = target.view(self.num_anchors * self.num_classes, self.grid_size, self.grid_size)

        return img, target






