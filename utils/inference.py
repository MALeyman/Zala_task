
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from torchvision.ops import nms
from PIL import Image
import os 
import numpy as np

def inference_model_custom(model, image_path, device, S=16, B=2, C=8, threshold=0.3, iou_thresh=0.3, image_size = 736):
    model.eval()

    # ==============    Преобразование изображения 
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # ===============   Предсказание 
    with torch.no_grad():
        preds = model(img_tensor)[0].cpu()  # (S, S, B*(5+C))

    img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    h, w = img_np.shape[:2]
    cell_size_x = w / S
    cell_size_y = h / S

    boxes = []
    confidences = []

    for y in range(S):
        for x in range(S):
            for b in range(B):
                base = b * (5 + C)
                conf = preds[y, x, base + 4].item()
                if conf < threshold:
                    continue

                px, py, pw, ph = preds[y, x, base:base+4]

                if not (0 <= px <= 1 and 0 <= py <= 1 and 0.01 <= pw <= 1.0 and 0.01 <= ph <= 1.0):
                    continue

                abs_x = (x + px) * cell_size_x
                abs_y = (y + py) * cell_size_y
                abs_w = pw * w
                abs_h = ph * h

                x1 = abs_x - abs_w / 2
                y1 = abs_y - abs_h / 2
                x2 = abs_x + abs_w / 2
                y2 = abs_y + abs_h / 2

                boxes.append([x1, y1, x2, y2])
                confidences.append(conf)

    if not boxes:
        print("Нет боксов выше порога.")
        return

    boxes = torch.tensor(boxes)
    scores = torch.tensor(confidences)

    keep = nms(boxes, scores, iou_thresh)

    # ===========    Отображение 
    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(img_np)

    for idx in keep:
        x1, y1, x2, y2 = boxes[idx]
        conf = scores[idx]

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{conf:.2f}", color='red', fontsize=9)

    ax.set_title("Найденные объекты")
    plt.axis('off')
    plt.show()









def plot_img(model, img_path):

    # 3. Переводим модель в eval режим
    model.eval()

    # 4. на CPU или GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 5. Загрузка изображения
    image = Image.open(img_path).convert('RGB')

    # 6. Преобразования 
    transform = transforms.Compose([
        transforms.Resize((736, 736)),                  # размер, на котором училась модель
        transforms.ToTensor(),                          # перевод в тензор [0,1], формат [C,H,W]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],  
        #                      std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image)  # [C, H, W]

    # 7. Добавляем batch dimension
    input_batch = input_tensor.unsqueeze(0)  # [1, C, H, W]

    # 8. Переносим на устройство
    input_batch = input_batch.to(device)
    model.to(device)

    # Предсказания:
    model.eval()
    with torch.no_grad():
        image_tensor = transforms.ToTensor()(image)  
        input_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(input_tensor)

    return image_tensor, outputs



def inference_model_grid_net(
    model,
    img_path,
    threshold=0.5,
    num_anchors=3,
    num_classes=8,
    class_names=None,
    max_entries=100,
    line_height=0.07
):
    """
    Визуализация выходов модели YOLO в формате [1, A*C, S, S]
    """

    image_tensor, outputs = plot_img(model, img_path)

    assert outputs.ndim == 4 and outputs.shape[0] == 1, "Ожидается выход модели: [1, A*C, S, S]"
    A, C = num_anchors, num_classes
    _, pred_dim, S, _ = outputs.shape
    assert pred_dim == A * C, f"Размер выхода {pred_dim} не соответствует {A}*{C}"

    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    H, W = image.shape[:2]

    # Переформатируем в [A, C, S, S]
    preds = outputs.view(1, A, C, S, S)[0]

    grid_y = np.linspace(0, H, S + 1, dtype=int)
    grid_x = np.linspace(0, W, S + 1, dtype=int)

    class_cells = []
    for gy in range(S):
        for gx in range(S):
            for a in range(A):
                for c in range(C):
                    score = torch.sigmoid(preds[a, c, gy, gx]).item()
                    if score > threshold:
                        x1 = grid_x[gx]
                        y1 = grid_y[gy]
                        x2 = grid_x[gx + 1]
                        y2 = grid_y[gy + 1]
                        class_cells.append((c, gy, gx, score, x1, y1, x2 - x1, y2 - y1))

    # Ограничим количество отображаемых записей
    class_cells = sorted(class_cells, key=lambda x: -x[3])[:max_entries]

    # Первая фигура: изображение с ячейками 
    fig_img, ax_img = plt.subplots(figsize=(12, 12))
    ax_img.imshow(image)
    ax_img.set_title('Обнаруженные ячейки')

    for x in grid_x:
        ax_img.axvline(x=x, color='white', linestyle='--', linewidth=0.2)
    for y in grid_y:
        ax_img.axhline(y=y, color='white', linestyle='--', linewidth=0.2)

    for c, gy, gx, score, x1, y1, w, h in class_cells:
        ax_img.add_patch(patches.Rectangle((x1, y1), w, h, edgecolor='lime', linewidth=1, fill=False))
        label = f"{c}: {score:.2f}"
        ax_img.text(x1 + 2, y1 + 12, label, color='yellow', fontsize=7)

    plt.tight_layout()
    plt.show()

    # Вторая фигура: список классов и координат 
    fig_height = max(1.5, line_height * len(class_cells) + 0.4)
    fig_text, ax_text = plt.subplots(figsize=(6, fig_height))
    ax_text.axis('off')
    ax_text.set_title('Классы, вероятности и ячейки')

    for idx, (c, gy, gx, score, *_ ) in enumerate(class_cells):
        name = class_names[c] if class_names else f"Класс {c}"
        ax_text.text(
            0.01, 1 - idx * line_height,
            f"{name:<12} -> Ячейка (Y={gy:02}, X={gx:02}) — Score: {score:.2f}",
            fontsize=9,
            transform=ax_text.transAxes
        )

    plt.tight_layout()
    plt.show()












