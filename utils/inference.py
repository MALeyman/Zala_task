
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from torchvision.ops import nms
from PIL import Image
import os 

def inference_model(model, image_path, device, S=16, B=2, C=8, threshold=0.3, iou_thresh=0.3, image_size = 736):
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
