'''  
Автор: Лейман М.А.
Дата создания: 17.07.2025

'''


import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torchvision.ops import box_iou
import torch, random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.patches as patches
import numpy as np
import torch






def yolo_loss(pred, target, B, C=8, lambda_coord=5.0, lambda_obj=1.0, lambda_noobj=0.1):
    """
        Функция потерь для модели, вычисляющая общую потерю (loss) для каждой ячейки сетки.

    Параметры:
    pred : torch.Tensor - Тензор с предсказаниями модели, размерностью [Bath, S, S, B*(5 + C)],
        где:
            Bath — размер батча,
            S — размер сетки,
            5 — количество параметров для каждого бокса (x, y, w, h, уверенность),
            C — количество классов.
    
    target : torch.Tensor - Тензор с целевыми значениями, аналогичный размерности pred.
        
    S : int, по умолчанию 16 - Размер сетки, на которую разбивается изображение.
        
    B : int, по умолчанию 2 - Количество боксов, которые предсказываются для каждой ячейки.
        
    C : int, по умолчанию 3 - Количество классов, которые модель может предсказать.
        
    lambda_coord : float, по умолчанию 5 - Коэффициент для усиления потерь по координатам (x, y, w, h).
        
    lambda_noobj : float, по умолчанию 0.5 - Коэффициент для усиления потерь для ячеек, где нет объектов.

    Описание:
    Функция рассчитывает суммарную потерю для каждой ячейки сетки, используя несколько компонентов потерь:
    
        1. Потери по координатам (loss_coord):
                Рассчитываются только для ячеек, содержащих объект.
                Ошибка в координатах (x, y) и размерности бокса (w, h) с коэффициентом lambda_coord.

        2. Потери для объектов (loss_obj):
                Для ячеек, где присутствует объект, рассчитывается ошибка уверенности (confidence).

        3. Потери для отсутствия объектов (loss_noobj):
                Для ячеек, где нет объектов, рассчитывается ошибка уверенности с коэффициентом lambda_noobj.

        4. Потери для классов (loss_class):
                Для ячеек с объектами рассчитывается ошибка по классам (one-hot encoding).
       
    Все потери суммируются для получения общего значения потерь.

    Выход:
        Общая потеря для всего батча, нормализованная на размер батча.
    """

    mse = nn.MSELoss(reduction='sum')
    bce = nn.BCEWithLogitsLoss(reduction='sum')  # для классов
    total_loss = 0.0
    obj_count = 0

    for b in range(B):
        start = b * (5 + C)
        
        obj_mask = target[..., start + 4] > 0
        noobj_mask = target[..., start + 4] == 0

        # Координаты x, y, w, h
        pred_box = pred[..., start:start + 4][obj_mask]
        target_box = target[..., start:start + 4][obj_mask]

        if pred_box.numel() > 0:
            loss_coord = lambda_coord * mse(pred_box, target_box)
        else:
            loss_coord = 0.0

        # Уверенность
        pred_obj = pred[..., start + 4][obj_mask]
        target_obj = target[..., start + 4][obj_mask]

        if pred_obj.numel() > 0:
            loss_obj = lambda_obj * mse(pred_obj, target_obj)
        else:
            loss_obj = 0.0

        pred_noobj = pred[..., start + 4][noobj_mask]
        target_noobj = target[..., start + 4][noobj_mask]
        loss_noobj = lambda_noobj * mse(pred_noobj, target_noobj)

        # Классы
        pred_class = pred[..., start + 5:start + 5 + C][obj_mask]
        target_class = target[..., start + 5:start + 5 + C][obj_mask]

        if pred_class.numel() > 0:
            loss_class = bce(pred_class, target_class)
        else:
            loss_class = 0.0

        total_loss += loss_coord + loss_obj + loss_noobj + loss_class
        obj_count += max(1, obj_mask.sum())  # защита от деления на 0

    return total_loss / obj_count  # нормализация по количеству объектов


def yolo_xywh_to_xyxy(box):
    # box: (..., 4)  ->  (..., 4)
    x_c, y_c, w, h = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    x1 = x_c - w * 0.5
    y1 = y_c - h * 0.5
    x2 = x_c + w * 0.5
    y2 = y_c + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)




def extract_boxes(preds, targets, S, B, C, device):
    """
    Извлекает предсказанные и истинные ограничивающие боксы, выбирая лучший предсказанный
    по IoU среди всех B боксов в ячейке, где есть объект.

    Аргументы:
        preds (Tensor): [batch_size, S, S, B*(5+C)]
        targets (Tensor): [batch_size, S, S, B*(5+C)]
        S (int): Размер сетки.
        B (int): Количество боксов на ячейку.
        C (int): Количество классов.
        device: torch.device.

    Возвращает:
        pred_boxes (Tensor): [N, 4]
        true_boxes (Tensor): [N, 4]
    """

    batch_size = preds.size(0)
    pred_boxes = []
    true_boxes = []

    for i in range(batch_size):
        for y in range(S):
            for x in range(S):
                # Проверяем, есть ли объект в этой ячейке (confidence > 0 в любом из B target-боксов)
                for b in range(B):
                    base = b * (5 + C)
                    if targets[i, y, x, base + 4] > 0:
                        true_box = targets[i, y, x, base:base+4]
                        break
                else:
                    continue  # если ни один b не был valid, то нет объекта в ячейке

                # Ищем лучший предсказанный бокс по IoU из всех B
                best_iou = 0
                best_pred_box = None
                for b in range(B):
                    base = b * (5 + C)
                    pred_box = preds[i, y, x, base:base+4]
                    iou = calculate_iou(pred_box.unsqueeze(0), true_box.unsqueeze(0))  # [1]
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_box = pred_box

                if best_pred_box is not None:
                    pred_boxes.append(best_pred_box)
                    true_boxes.append(true_box)

    if len(pred_boxes) == 0:
        return None, None

    pred_boxes = torch.stack(pred_boxes).to(device)
    true_boxes = torch.stack(true_boxes).to(device)

    return pred_boxes, true_boxes



def calculate_iou_batch(boxes_preds, boxes_labels, eps=1e-6):
    """
    Вычисляет IoU по батчу предсказанных и истинных боксов.
    
    Формат входа:
        boxes_preds: Tensor формы (N, 4) — предсказанные боксы.
        boxes_labels: Tensor формы (N, 4) — истинные боксы.
    Формат бокса: (x_center, y_center, width, height)

    Описание:
        1. Преобразует YOLO-формат (центр + размеры) в координаты углов (xmin, ymin, xmax, ymax).
        2. Рассчитывает пересечение по каждой координате.
        3. Площадь пересечения = ширина * высота, с clamp(0), чтобы избежать отрицательных значений.
        4. Вычисляет площади каждого бокса.
        5. IoU = пересечение / объединение (добавлен небольшой epsylon  для избежания деления на 0).
    
    Возвращает:
        Тензор размера (N,) — значения IoU для каждой пары предсказание/истина в батче.
    """


    # Преобразуем в (x1, y1, x2, y2)
    pred_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
    pred_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
    pred_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
    pred_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2

    label_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
    label_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
    label_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
    label_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    # Пересечение
    inter_x1 = torch.max(pred_x1, label_x1)
    inter_y1 = torch.max(pred_y1, label_y1)
    inter_x2 = torch.min(pred_x2, label_x2)
    inter_y2 = torch.min(pred_y2, label_y2)
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Площади
    area_pred = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    area_label = (label_x2 - label_x1) * (label_y2 - label_y1)

    union = area_pred + area_label - inter_area + eps
    iou = inter_area / union

    # Обработка крайних случаев
    both_empty = (area_pred == 0) & (area_label == 0)
    iou[both_empty] = 1.0

    return iou


def calculate_iou(box1, box2):  # не используется
    """
    Вычисляет IoU (Intersection over Union) между двумя bboxes (bounding boxes).
    
    Формат входа:
        box1, box2: Тензоры произвольной размерности с последним измерением 4
            (x_center, y_center, width, height) — YOLO-формат.
    
    Этапы:
        1. Преобразование центра и размеров в координаты углов (xmin, ymin, xmax, ymax).
        2. Вычисление координат пересечения (inter_x1, inter_y1, inter_x2, inter_y2).
        3. Площадь пересечения: ширина * высота.
        4. Площадь объединения: area1 + area2 - inter_area.
        5. Итоговое значение IoU = inter_area / union_area.

    Возвращает:
        IoU между каждым соответствующим парой box1 и box2.
    """
    # box: [x_center, y_center, w, h]
    x1_min = box1[..., 0] - box1[..., 2] / 2
    y1_min = box1[..., 1] - box1[..., 3] / 2
    x1_max = box1[..., 0] + box1[..., 2] / 2
    y1_max = box1[..., 1] + box1[..., 3] / 2

    x2_min = box2[..., 0] - box2[..., 2] / 2
    y2_min = box2[..., 1] - box2[..., 3] / 2
    x2_max = box2[..., 0] + box2[..., 2] / 2
    y2_max = box2[..., 1] + box2[..., 3] / 2

    inter_x1 = torch.max(x1_min, x2_min)
    inter_y1 = torch.max(y1_min, y2_min)
    inter_x2 = torch.min(x1_max, x2_max)
    inter_y2 = torch.min(y1_max, y2_max)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area + 1e-6
    iou = inter_area / union_area
    return iou




def draw_boxes(
        img_tensor,
        pred_tensor,
        true_tensor,
        S=16, B=2, C=8,
        threshold=0.5,
        class_names=None        # список имён или None
    ):
    """
    Визуализация GT-боксов (зелёные) и предсказаний (красные).
    Показывает confident-карты и подписывает номер/имя класса.
    """

    # подготовка изображения 
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    h, w = img.shape[:2]

    # карты уверенности
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    gt_map = true_tensor[..., 4].cpu().numpy()
    im0 = axes[0].imshow(gt_map, cmap='hot', interpolation='nearest')
    axes[0].set_title("Target confidence")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    pred_map = np.zeros((S, S))
    for y in range(S):
        for x in range(S):
            for b in range(B):
                base = b * (5 + C)
                conf = pred_tensor[y, x, base + 4].item()
                pred_map[y, x] = max(pred_map[y, x], conf)

    im1 = axes[1].imshow(pred_map, cmap='hot', interpolation='nearest')
    axes[1].set_title("Pred confidence")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    # отрисовка боксов 
    cell_w = w / S
    cell_h = h / S
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    # GT (зелёные) 
    for y in range(S):
        for x in range(S):
            for b in range(B):
                base = b * (5 + C)
                if true_tensor[y, x, base + 4] > 0:
                    tx, ty, tw, th = true_tensor[y, x, base:base+4]
                    cls_vec = true_tensor[y, x, base + 5 : base + 5 + C]
                    cls_id  = int(cls_vec.argmax())
                    label   = class_names[cls_id] if class_names else str(cls_id)

                    cx = (x + tx.item()) * cell_w
                    cy = (y + ty.item()) * cell_h
                    bw = tw.item() * w
                    bh = th.item() * h

                    rect = patches.Rectangle((cx - bw/2, cy - bh/2),
                                             bw, bh,
                                             linewidth=2, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                    # ax.text(cx - bw/2, cy - bh/2 - 4,
                    #         label, color='lime', fontsize=8, weight='bold')

    # предсказания (красные) 
    for y in range(S):
        for x in range(S):
            best_conf = 0.0
            best_box  = None
            best_cls  = None
            for b in range(B):
                base = b * (5 + C)
                conf = pred_tensor[y, x, base + 4].item()
                if conf > best_conf and conf >= threshold:
                    px, py, pw, ph = pred_tensor[y, x, base:base+4]
                    if not (0 <= px <= 1 and 0 <= py <= 1 and 0.01 <= pw <= 1.0 and 0.01 <= ph <= 1.0):
                        continue
                    cls_vec = pred_tensor[y, x, base + 5 : base + 5 + C]
                    best_cls = int(cls_vec.argmax())
                    best_conf = conf
                    best_box  = (px.item(), py.item(), pw.item(), ph.item())

            if best_box:
                px, py, pw, ph = best_box
                cx = (x + px) * cell_w
                cy = (y + py) * cell_h
                bw = pw * w
                bh = ph * h
                if bw < 3 or bh < 3 or bw > w or bh > h:
                    continue

                label = class_names[best_cls] if class_names else str(best_cls)
                rect  = patches.Rectangle((cx - bw/2, cy - bh/2),
                                          bw, bh,
                                          linewidth=1.5, edgecolor='red',
                                          linestyle='--', facecolor='none')
                ax.add_patch(rect)
                ax.text(cx - bw/2, cy - bh/2 - 4,
                        f"{label}:{best_conf:.2f}",
                        color='red', fontsize=8, weight='bold')

    # сетка
    for i in range(1, S):
        ax.axhline(i * cell_h, color='gray', linestyle=':', linewidth=0.5)
        ax.axvline(i * cell_w, color='gray', linestyle=':', linewidth=0.5)

    ax.set_title("Green = GT,  Red = Pred")
    ax.axis('off')
    plt.show()





@torch.no_grad()
def val_epoch(model, loader, loss_fn,
              device, S, B, C,
              iou_thr=0.5,     # для TP-решения
              conf_thr=0.35):  # каких предсказаний брать
    """
    Возвращает: avg_loss, avg_iou, precision, recall
    """
    model.eval()
    total_loss, total_iou, iou_cnt = 0.0, 0.0, 0
    TP = FP = FN = 0


    loop = tqdm(loader, desc="Валидация", leave=False)
    for imgs, targets in loop:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, targets, B, C).item()
        total_loss += loss

        #  IoU 
        pb, tb = extract_boxes(preds, targets, S, B, C, device)
        if pb is not None:
            total_iou += calculate_iou_batch(pb, tb).sum().item()
            iou_cnt   += pb.size(0)

        #  Precision / Recall 
        bs = imgs.size(0)
        for b_idx in range(bs):
            # GT-боксы (xywh) для одной картинки
            _, gt_boxes = extract_boxes(preds[b_idx:b_idx+1],
                                        targets[b_idx:b_idx+1],
                                        S, B, C, device)
            gt_boxes = gt_boxes if gt_boxes is not None else torch.empty((0,4), device=device)

            # Все предсказания с conf >= conf_thr
            pred_list = []
            one_pred = preds[b_idx]
            for y in range(S):
                for x in range(S):
                    for b in range(B):
                        base = b*(5+C)
                        conf = one_pred[y, x, base+4]
                        if conf >= conf_thr:
                            pred_list.append(one_pred[y, x, base:base+4])
            pred_boxes = torch.stack(pred_list) if pred_list else torch.empty((0,4), device=device)

            # перевод в (x1,y1,x2,y2)
            pb_xyxy = yolo_xywh_to_xyxy(pred_boxes)
            gt_xyxy = yolo_xywh_to_xyxy(gt_boxes)

            # подсчёт TP / FP / FN
            if pb_xyxy.numel() == 0 and gt_xyxy.numel() == 0:
                continue
            if pb_xyxy.numel() == 0:
                FN += gt_xyxy.size(0);                 continue
            if gt_xyxy.numel() == 0:
                FP += pb_xyxy.size(0);                 continue

            ious = box_iou(pb_xyxy, gt_xyxy)           # (N_pred, N_gt)
            max_iou_pred, gt_idx = ious.max(dim=1)

            matched_gt = torch.zeros(gt_xyxy.size(0), dtype=torch.bool, device=device)
            for miou, g in zip(max_iou_pred, gt_idx):
                if miou >= iou_thr and not matched_gt[g]:
                    TP += 1
                    matched_gt[g] = True
                else:
                    FP += 1
            FN += (~matched_gt).sum().item()
        loop.set_postfix(val_loss=loss)

    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    mean_iou  = total_iou / (iou_cnt + 1e-6)
    avg_loss  = total_loss / len(loader)


    # Визуализация
    idx = random.randint(0, imgs.size(0) - 1)
    clear_output(wait=True)
    draw_boxes(imgs[idx].cpu(), preds[idx].cpu(), targets[idx].cpu(), S, B, C)
    
    return avg_loss, mean_iou, precision, recall



def train_epoch(model, loader, optimizer, loss_fn, device, S, B, C):
    """  
        Тренировка
    """
    model.train()
    total_loss = 0
    total_iou = 0
    iou_count = 0

    loop = tqdm(loader, desc="Тренировка", leave=False)
    for imgs, targets in loop:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        
        loss = loss_fn(preds, targets, B, C)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_boxes, true_boxes = extract_boxes(preds, targets, S, B, C, device)
        if pred_boxes is not None and true_boxes is not None:
            ious = calculate_iou_batch(pred_boxes, true_boxes)
            total_iou += ious.sum().item()
            iou_count += ious.numel()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    avg_iou = total_iou / iou_count if iou_count > 0 else 0
    return avg_loss, avg_iou



def train_mobile_net(model, train_loader_1, train_loader_2, train_loader_3, train_loader_4, val_loader, device, lr_init=0.001, epochs=10, S=16, B=2, C=8, save_path="models_custom/checkpoints", name_model="best_model.pth"):

    print("Использование устройства :", device)
    os.makedirs(save_path, exist_ok=True)
    train_losses = []
    val_losses = []
    val_ious = []
    train_ious = []
    patience = 10
    wait = 0
    best_val_loss = float("inf")

    optimizer = optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.3)
    for epoch in range(epochs):
        print(f"\n[Эпоха {epoch+1}/{epochs}]")



        # Обучение модели
        if (epoch + 1) < 2:
            train_loss, train_iou = train_epoch(model, val_loader, optimizer, yolo_loss, device, S, B, C)

        elif (epoch + 1) < 4:
            train_loss, train_iou = train_epoch(model, train_loader_1, optimizer, yolo_loss, device, S, B, C)
        elif (epoch + 1) % 2 == 0:
           train_loss, train_iou = train_epoch(model, train_loader_3, optimizer, yolo_loss, device, S, B, C)
         
        else:
            train_loss, train_iou = train_epoch(model, train_loader_4, optimizer, yolo_loss, device, S, B, C)
        

        val_loss, val_iou, precision, recall = val_epoch(model, val_loader, yolo_loss, device, S, B, C)
        scheduler.step()

        print(f"Эпоха {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Train IoU={train_iou:.4f} | VAL IoU={val_iou:.4f}")
        print(f"Эпоха {epoch+1}: Val precision={precision:.4f} | VAL recall={recall:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

            # =========   Сохраняем модель, если улучшение 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), os.path.join(save_path, name_model))
            print("Сохранена лучшая модель.")
        else:
            wait += 1
            print(f"Улучшений нет. Patience: {wait}/{patience}")
            if wait >= patience:
                print("Обучение остановлено.")
                break


    # Графики
    plt.figure(figsize=(12, 4))

    #  Train Val Loss 
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Train  Val IoU 
    plt.subplot(1, 2, 2)
    plt.plot(train_ious, label='Train IoU', color='blue')
    plt.plot(val_ious, label='Val IoU', color='green')
    plt.title("IoU")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()

    plt.tight_layout()
    plt.show()               



def calculate_precision_recall(preds, targets, num_anchors=3, num_classes=8, threshold=0.5):
    """
    Вычисляет precision и recall по каждому классу для anchor-based multi-label классификации ячеек.

    Аргументы:
        preds:   [batch_size, B*C, S, S] — логиты
        targets: [batch_size, B*C, S, S] — бинарные метки
        threshold: float — порог для перевода логитов в бинарные предсказания

    Возвращает:
        precisions: list[float], по num_classes
        recalls:    list[float], по num_classes
    """
    B, BC, S, _ = preds.shape
    assert BC == num_anchors * num_classes, f"Ожидалось B*C каналов, но получили {BC}"

    preds = preds.view(B, num_anchors, num_classes, S, S)
    targets = targets.view(B, num_anchors, num_classes, S, S)

    # сигмоида + бинаризация
    pred_mask = torch.sigmoid(preds) > threshold
    target_mask = targets.bool()

    # агрегируем по якорям — если хотя бы один якорь предсказал класс в ячейке
    pred_per_cell = pred_mask.any(dim=1)     # [B, C, S, S]
    target_per_cell = target_mask.any(dim=1) # [B, C, S, S]

    precisions, recalls = [], []

    for c in range(num_classes):
        pred_flat = pred_per_cell[:, c].flatten()
        true_flat = target_per_cell[:, c].flatten()

        TP = (pred_flat & true_flat).sum().item()
        FP = (pred_flat & ~true_flat).sum().item()
        FN = (~pred_flat & true_flat).sum().item()

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)

        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls



def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_prec'], label='Train Precision')
    plt.plot(epochs, history['val_prec'], label='Val Precision')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_recall'], label='Train Recall')
    plt.plot(epochs, history['val_recall'], label='Val Recall')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.show()

 

def train_model_grid_net(model,
    train_loader_1,
    train_loader_2,
    train_loader_3,
    train_loader_4, 
    val_loader, 
    optimizer, scheduler, loss_fn, metric_fn,
                num_epochs=1, device='cuda', save_path='best_model.pth'):

    model = model.to(device)
    history = {
        'train_loss': [], 'val_loss': [],
        'train_prec': [], 'val_prec': [],
        'train_recall': [], 'val_recall': []
    }

    best_val_recall = 0.0  # для отслеживания лучшего recall

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_prec, train_recall = 0, 0, 0

        # Обучение модели
        if (epoch + 1) < 2:
            train_loader = train_loader_2

        elif (epoch + 1) < 4:
            train_loader = train_loader_1
        elif (epoch + 1) % 2 == 0:
            train_loader = train_loader_4
         
        else:
            train_loader = train_loader_3
        


        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for images, targets in loop:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)

    
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            prec, recall = metric_fn(outputs.detach(), targets)

            train_loss += loss.item()

            avg_prec = sum(prec) / len(prec)
            avg_recall = sum(recall) / len(recall)

            train_prec += avg_prec
            train_recall += avg_recall


            loop.set_postfix(loss=loss.item(), prec=prec, recall=recall)

        n_train = len(train_loader)
        history['train_loss'].append(train_loss / n_train)
        history['train_prec'].append(train_prec / n_train)
        history['train_recall'].append(train_recall / n_train)

        # Валидация
        model.eval()
        val_loss, val_prec, val_recall = 0, 0, 0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Validation [{epoch+1}/{num_epochs}]", leave=False)
            for images, targets in val_loop:

                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, targets)
                prec, recall = metric_fn(outputs, targets)

                val_loss += loss.item()

                avg_prec = sum(prec) / len(prec)
                avg_recall = sum(recall) / len(recall)

                val_prec += avg_prec
                val_recall += avg_recall

                val_loop.set_postfix(loss=loss.item(), prec=prec, recall=recall)

        n_val = len(val_loader)
        history['val_loss'].append(val_loss / n_val)
        history['val_prec'].append(val_prec / n_val)
        history['val_recall'].append(val_recall / n_val)

        print(f"\n Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | "
              f"Val Prec: {history['val_prec'][-1]:.4f} | Val Recall: {history['val_recall'][-1]:.4f}\n")
        recall_avg = val_recall / n_val
        # Сохраняем лучшую модель по recall
        if recall_avg > best_val_recall:
            best_val_recall = recall_avg
            torch.save(model.state_dict(), save_path)
            print(f"Сохранена лучшая модель {epoch+1} с val recall {best_val_recall:.4f}")

        scheduler.step()

    plot_metrics(history)
    return history



class YOLOSimplifiedLoss(nn.Module):
    def __init__(self, S, B, C, lambda_obj=1.0, lambda_noobj=0.2, lambda_cls=1.0):
        """
        S — размер сетки
        B — количество якорей на ячейку
        C — количество классов
        """
        super(YOLOSimplifiedLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls

        self.bce = nn.BCEWithLogitsLoss(reduction='sum')  # для objectness и классов

    def forward(self, predictions, targets):
        """
        predictions: [bath, B*C, S, S] — логиты
        targets:     [bath, B*C, S, S] — бинарные метки (0 или 1)
        """
        batch_size = predictions.size(0)

        # [bath, B, C, S, S]
        preds = predictions.view(batch_size, self.B, self.C, self.S, self.S)
        targs = targets.view(batch_size, self.B, self.C, self.S, self.S)

        # Objectness mask (есть ли объект в позиции)
        obj_mask = targs > 0.5  # [bath, B, C, S, S]
        noobj_mask = ~obj_mask

        # Objectness loss
        obj_loss = self.bce(preds[obj_mask], targs[obj_mask])
        noobj_loss = self.bce(preds[noobj_mask], targs[noobj_mask])

        total_loss = self.lambda_obj * obj_loss + self.lambda_noobj * noobj_loss
        total_loss = total_loss / batch_size

        return total_loss