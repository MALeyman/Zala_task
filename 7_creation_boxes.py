"""
    Автор: Лейман М.А.
    Дата создания: 10.04.2025
"""

"""  
    ===== Горячие клавиши: =====

    m       # Выбор класса
    q       # Выход
    d       # Пропустить изображение

    s       # сохранять текущий кадр в датасет train/
    a       # сохранять текущий кадр в датасет test/
    z       # отменять последнее изменение (undo) — возврат «один бокс назад» 

"""



import os
import tkinter as tk
from tkinter import Canvas, filedialog
from PIL import Image, ImageTk
import cv2
import glob
from tkinter import simpledialog
import shutil
import random

# Получаем текущую рабочую папку  
current_directory = os.getcwd()  


path_dataset = os.path.join(current_directory, "devsTraining/zala_task/dataset")

# # image_2 = 'dataset_VEDAI/image_original/'  
# image_2 = 'dataset_DOTA/images_jpeg/'
# image_2 =  "dataset_mobileNet/dataset_0/image/"

# images_path_1 = os.path.join(path_dataset, image_2) 

# # labels_2 = 'dataset_UAVOD/datasets_original/target_yolo/'   
# labels_2 = 'dataset_DOTA/cleaned_labels_2/' 
# labels_2 = 'dataset_mobileNet/dataset_0/target/' 

# labels_path = os.path.join(path_dataset, labels_2)  

# print(images_path_1)

image_3 = 'custom_dataset/'   
images_path_out = os.path.join(path_dataset, image_3, "image") 

labels_3 = 'custom_dataset/'   
labels_path_out = os.path.join(path_dataset, labels_3, "target")  

print(images_path_out)
print(labels_path_out)


#  Папки 
images_path="/home/maksim/develops/python/devsTraining/zala_task/dataset/dataset_DOTA/set/image/train/"
labels_path="/home/maksim/develops/python/devsTraining/zala_task/dataset/dataset_DOTA/set/target/train/"

# images_path="/home/maksim/develops/python/devsTraining/zala_task/dataset/dataset_full_1/images/train/"
# labels_path="/home/maksim/develops/python/devsTraining/zala_task/dataset/dataset_full_1/labels/train/"

# images_path="/home/maksim/develops/python/devsTraining/zala_task/dataset/custom_dataset/mix_out/images/"
# labels_path="/home/maksim/develops/python/devsTraining/zala_task/dataset/custom_dataset/mix_out/labels/"


images_path="/home/maksim/develops/python/devsTraining/zala_task/dataset/dataset_full_1/images/val/"
labels_path="/home/maksim/develops/python/devsTraining/zala_task/dataset/dataset_full_1/labels/val/"
print(images_path)

output_images_dir = images_path_out
output_labels_dir = labels_path_out

# output_images_dir = images_path
# output_labels_dir = labels_path

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

#  каталоги назначения  
train_img_dir  = os.path.join(output_images_dir, "train")
train_lbl_dir  = os.path.join(output_labels_dir, "train")
test_img_dir   = os.path.join(output_images_dir, "test")
test_lbl_dir   = os.path.join(output_labels_dir, "test")

# #  каталоги назначения  
# train_img_dir  = images_path
# train_lbl_dir  = labels_path

for d in (train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir):
    os.makedirs(d, exist_ok=True)


#  стек истории для undo  
history = []           # хранит tuple("add"/"del", box)


#  Переменные 
image_files = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
current_index = 0
boxes = []
new_boxes = []
drawing = False
start_x = start_y = 0
current_rect = None
current_class_id = 0
img = None
tk_img = None

#  Окно
root = tk.Tk()
root.title("YOLO Аннотации")

frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

canvas = Canvas(frame, cursor="cross", bg="black")
canvas.grid(row=0, column=0, sticky="nsew")

x_scroll = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
x_scroll.grid(row=1, column=0, sticky="ew")

y_scroll = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
y_scroll.grid(row=0, column=1, sticky="ns")

frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)

canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)


def load_image(index):
    global img, tk_img, boxes, new_boxes
    if index >= len(image_files):
        print("Все изображения размечены.")
        root.quit()
        return

    img_path = image_files[index]
    print(f"\nЗагрузка: {img_path}")
    filename = os.path.basename(img_path)
    root.title(f"YOLO Аннотации — {filename}")

    cv_img = cv2.imread(img_path)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv_img)
    tk_img = ImageTk.PhotoImage(img)

    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas.config(scrollregion=canvas.bbox(tk.ALL))

    boxes.clear()
    new_boxes.clear()

    # Загрузка YOLO разметки
    basename = os.path.splitext(os.path.basename(img_path))[0]
    label_file = os.path.join(labels_path, f"{basename}.txt")
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = map(float, parts)
                w_img, h_img = img.size
                x1 = int((x - w / 2) * w_img)
                y1 = int((y - h / 2) * h_img)
                x2 = int((x + w / 2) * w_img)
                y2 = int((y + h / 2) * h_img)
                boxes.append((int(cls), x1, y1, x2, y2))
    else:
        print("Файл аннотации не найден — начнём разметку с нуля.")

    redraw()



def redraw():
    canvas.delete("box")

    # =============   Нарисовать сетку для удобства 
    grid_spacing = 100  # Размер клетки сетки в пикселях
    w_img, h_img = img.size

    for x in range(0, w_img, grid_spacing):
        canvas.create_line(x, 0, x, h_img, fill="gray", dash=(2, 4), tags="box")
    for y in range(0, h_img, grid_spacing):
        canvas.create_line(0, y, w_img, y, fill="gray", dash=(2, 4), tags="box")

    # ===========   Нарисовать все текущие боксы
        CLASS_COLORS = {
            0: "blue",
            1: "green",
            2: "red",
            3: "orange",
            4: "purple",
            5: "red",
            6: "red",
            7: "red",
        }
        DEFAULT_COLOR = "gray"

    for cls, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="box")
        canvas.create_text(x1 + 4, y1 - 17, text=str(cls), anchor="nw",
                           fill=color, font=("Arial", 14, "bold"), tags="box")

    # for cls, x1, y1, x2, y2 in boxes:
    #     color = "blue" if cls == 0 else "green"
    #     canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="box")
    #     canvas.create_text(x1 + 4, y1 - 17, text=str(cls), anchor="nw",
    #                     fill=color, font=("Arial", 14, "bold"), tags="box")

        # canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2, tags="box")
        # canvas.create_text(x1 + 4, y1 - 17, text=str(cls), anchor="nw",
        #                    fill="green", font=("Arial", 14, "bold"), tags="box")




def on_mouse_down(event):
    global drawing, start_x, start_y, current_rect
    drawing = True
    start_x = canvas.canvasx(event.x)
    start_y = canvas.canvasy(event.y)
    current_rect = canvas.create_rectangle(start_x, start_y, start_x, start_y,
                                           outline="blue", width=1, tags="box")

def on_mouse_move(event):
    if not drawing or current_rect is None:
        return
    cur_x = canvas.canvasx(event.x)
    cur_y = canvas.canvasy(event.y)
    canvas.coords(current_rect, start_x, start_y, cur_x, cur_y)

def on_mouse_up(event):
    global drawing, current_rect
    if not drawing or current_rect is None:
        return
    drawing = False
    end_x = canvas.canvasx(event.x)
    end_y = canvas.canvasy(event.y)
    x1, y1 = int(min(start_x, end_x)), int(min(start_y, end_y))
    x2, y2 = int(max(start_x, end_x)), int(max(start_y, end_y))
    boxes.append((current_class_id, x1, y1, x2, y2))
    new_boxes.append((current_class_id, x1, y1, x2, y2))
    history.append(("add", boxes[-1])) 
    redraw()
    current_rect = None

# undo-функция 
def undo_last(event=None):
    if not history:
        print("История пуста.")
        return
    action, box = history.pop()
    if action == "add":
        # убираем последнюю добавленную
        boxes.pop()                 # всегда последняя
    else:  # "del"
        boxes.append(box)           # возвращаем удалённую
    redraw()


def on_right_click(event):
    x = canvas.canvasx(event.x)
    y = canvas.canvasy(event.y)
    for i in range(len(boxes) - 1, -1, -1):
        cls, x1, y1, x2, y2 = boxes[i]
        if x1 <= x <= x2 and y1 <= y <= y2:
            removed = boxes.pop(i)
            history.append(("del", removed)) 
            redraw()
            break

def yolo_format(x1, y1, x2, y2, cls):
    w_img, h_img = img.size
    x_center = ((x1 + x2) / 2) / w_img
    y_center = ((y1 + y2) / 2) / h_img
    w = abs(x2 - x1) / w_img
    h = abs(y2 - y1) / h_img
    return f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

def change_class_id():
    global current_class_id
    new_id = simpledialog.askstring("Выбор класса", "Введите новый class_id:")
    if new_id is not None:
        try:
            current_class_id = int(new_id)
            print(f"Теперь размечаем класс: {current_class_id}")
        except ValueError:
            print("Некорректный ввод.")


#  универсальный save() 
def save_current(to_train=True):
    if not img:
        return
    img_file   = os.path.basename(image_files[current_index])
    lbl_file   = os.path.splitext(img_file)[0] + ".txt"

    if to_train:
        img_dir, lbl_dir = train_img_dir, train_lbl_dir
    else:
        img_dir, lbl_dir = test_img_dir, test_lbl_dir

    # сохранение
    img.save(os.path.join(img_dir, img_file))
    with open(os.path.join(lbl_dir, lbl_file), "w") as f:
        for cls, x1, y1, x2, y2 in boxes:
            f.write(yolo_format(x1, y1, x2, y2, cls) + "\n")

    print(f"Сохранено в {'train' if to_train else 'test'}: {img_file}")


def skip_image():
    global current_index
    current_index += 1
    load_image(current_index)

def save_and_next(to_train=True, event=None):
    global current_index
    if not img:        # если ещё ничего не загружено
        return
    # 1) сохраняем
    save_current(to_train)
    # 2) увеличиваем индекс
    current_index += 1
    # 3) загружаем следующую картинку
    load_image(current_index)


#   Привязка событий
canvas.bind("<Button-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_move)
canvas.bind("<ButtonRelease-1>", on_mouse_up)
canvas.bind("<Button-3>", on_right_click)

# биндинги клавиш 
root.bind("m", lambda e: change_class_id()) # Выбор класса
root.bind("q", lambda e: root.quit())       # Выход
root.bind("d", lambda e: skip_image())      # Пропустить изображение

root.bind("s", lambda e: save_and_next(True))   # в train/
root.bind("a", lambda e: save_and_next(False))  # в test/
root.bind("z", undo_last)        # Undo


load_image(current_index)
root.mainloop()


