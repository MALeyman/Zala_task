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
from tkinter import Canvas, filedialog, simpledialog
from PIL import Image, ImageTk
import cv2
import glob
from tkinter import simpledialog

class YOLOAnnotatorApp:
    CLASS_COLORS = {
        0: "blue", 1: "green", 2: "red", 3: "orange", 4: "purple",
        5: "red", 6: "red", 7: "red"
    }
    DEFAULT_COLOR = "gray"
    GRID_SPACING = 100

    def __init__(self,
                 images_path,
                 labels_path,
                 output_images_dir,
                 output_labels_dir,
                 window_title="YOLO Аннотации"):
        self.images_path = images_path
        self.labels_path = labels_path
        self.output_images_dir = output_images_dir
        self.output_labels_dir = output_labels_dir
        self.window_title = window_title

        # Создаём директории
        self.train_img_dir = os.path.join(output_images_dir, "train")
        self.train_lbl_dir = os.path.join(output_labels_dir, "train")
        self.test_img_dir  = os.path.join(output_images_dir, "test")
        self.test_lbl_dir  = os.path.join(output_labels_dir, "test")
        for d in (self.train_img_dir, self.train_lbl_dir, self.test_img_dir, self.test_lbl_dir):
            os.makedirs(d, exist_ok=True)

        self.history = []
        self.image_files = sorted(
            glob.glob(os.path.join(self.images_path, "*.jpg")))
        self.current_index = 0
        self.boxes = []
        self.new_boxes = []
        self.drawing = False
        self.start_x = self.start_y = 0
        self.current_rect = None
        self.current_class_id = 0
        self.img = None
        self.tk_img = None

    def run(self):
        self.root = tk.Tk()
        self.root.title(self.window_title)
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = Canvas(frame, cursor="cross", bg="black")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        x_scroll = tk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        y_scroll = tk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)

        # Bindings
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.root.bind("m", lambda e: self.change_class_id())
        self.root.bind("q", lambda e: self.root.quit())
        self.root.bind("d", lambda e: self.skip_image())
        self.root.bind("s", lambda e: self.save_and_next(True))
        self.root.bind("a", lambda e: self.save_and_next(False))
        self.root.bind("z", lambda e: self.undo_last())
        self.load_image(self.current_index)
        self.root.mainloop()

    def load_image(self, index):
        if index >= len(self.image_files):
            print("Все изображения размечены.")
            self.root.quit()
            return

        img_path = self.image_files[index]
        print(f"\nЗагрузка: {img_path}")
        filename = os.path.basename(img_path)
        self.root.title(f"{self.window_title} — {filename}")

        cv_img = cv2.imread(img_path)
        if cv_img is None:
            print("Ошибка загрузки изображения:", img_path)
            return
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.img = Image.fromarray(cv_img)
        self.tk_img = ImageTk.PhotoImage(self.img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        self.boxes.clear()
        self.new_boxes.clear()

        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_file = os.path.join(self.labels_path, f"{basename}.txt")
        if os.path.exists(label_file):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x, y, w, h = map(float, parts)
                    w_img, h_img = self.img.size
                    x1 = int((x - w / 2) * w_img)
                    y1 = int((y - h / 2) * h_img)
                    x2 = int((x + w / 2) * w_img)
                    y2 = int((y + h / 2) * h_img)
                    self.boxes.append((int(cls), x1, y1, x2, y2))
        else:
            print("Файл аннотации не найден — начнём разметку с нуля.")

        self.redraw()

    def redraw(self):
        self.canvas.delete("box")
        if not self.img:
            return
        w_img, h_img = self.img.size
        for x in range(0, w_img, self.GRID_SPACING):
            self.canvas.create_line(x, 0, x, h_img, fill="gray", dash=(2, 4), tags="box")
        for y in range(0, h_img, self.GRID_SPACING):
            self.canvas.create_line(0, y, w_img, y, fill="gray", dash=(2, 4), tags="box")
        for cls, x1, y1, x2, y2 in self.boxes:
            color = self.CLASS_COLORS.get(cls, self.DEFAULT_COLOR)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="box")
            self.canvas.create_text(x1 + 4, y1 - 17, text=str(cls), anchor="nw",
                                   fill=color, font=("Arial", 14, "bold"), tags="box")

    def on_mouse_down(self, event):
        self.drawing = True
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="blue", width=1, tags="box"
        )

    def on_mouse_move(self, event):
        if not self.drawing or self.current_rect is None:
            return
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.current_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_mouse_up(self, event):
        if not self.drawing or self.current_rect is None:
            return
        self.drawing = False
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        x1, y1 = int(min(self.start_x, end_x)), int(min(self.start_y, end_y))
        x2, y2 = int(max(self.start_x, end_x)), int(max(self.start_y, end_y))
        self.boxes.append((self.current_class_id, x1, y1, x2, y2))
        self.new_boxes.append((self.current_class_id, x1, y1, x2, y2))
        self.history.append(("add", self.boxes[-1]))
        self.redraw()
        self.current_rect = None

    def on_right_click(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        for i in range(len(self.boxes) - 1, -1, -1):
            cls, x1, y1, x2, y2 = self.boxes[i]
            if x1 <= x <= x2 and y1 <= y <= y2:
                removed = self.boxes.pop(i)
                self.history.append(("del", removed))
                self.redraw()
                break

    def undo_last(self):
        if not self.history:
            print("История пуста.")
            return
        action, box = self.history.pop()
        if action == "add":
            self.boxes.pop()
        else:  # "del"
            self.boxes.append(box)
        self.redraw()

    def yolo_format(self, x1, y1, x2, y2, cls):
        w_img, h_img = self.img.size
        x_center = ((x1 + x2) / 2) / w_img
        y_center = ((y1 + y2) / 2) / h_img
        w = abs(x2 - x1) / w_img
        h = abs(y2 - y1) / h_img
        return f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

    def change_class_id(self):
        new_id = simpledialog.askstring("Выбор класса", "Введите новый class_id:")
        if new_id is not None:
            try:
                self.current_class_id = int(new_id)
                print(f"Теперь размечаем класс: {self.current_class_id}")
            except ValueError:
                print("Некорректный ввод.")

    def save_current(self, to_train=True):
        if not self.img:
            return
        img_file = os.path.basename(self.image_files[self.current_index])
        lbl_file = os.path.splitext(img_file)[0] + ".txt"

        if to_train:
            img_dir, lbl_dir = self.train_img_dir, self.train_lbl_dir
        else:
            img_dir, lbl_dir = self.test_img_dir, self.test_lbl_dir

        self.img.save(os.path.join(img_dir, img_file))
        with open(os.path.join(lbl_dir, lbl_file), "w") as f:
            for cls, x1, y1, x2, y2 in self.boxes:
                f.write(self.yolo_format(x1, y1, x2, y2, cls) + "\n")

        print(f"Сохранено в {'train' if to_train else 'test'}: {img_file}")

    def skip_image(self):
        self.current_index += 1
        self.load_image(self.current_index)

    def save_and_next(self, to_train=True):
        if not self.img:
            return
        self.save_current(to_train)
        self.current_index += 1
        self.load_image(self.current_index)


# ==========  ВЫЗВАТЬ 
if __name__ == '__main__':

    app = YOLOAnnotatorApp(
        images_path="/home/maksim/develops/python/Zala_task/Zala_task/data/dataset/dataset_full_1/images/train/",
        labels_path="/home/maksim/develops/python/Zala_task/Zala_task/data/dataset/dataset_full_1/target/train/",
        output_images_dir="/home/maksim/develops/python/Zala_task/Zala_task/data/output_images",
        output_labels_dir="/home/maksim/develops/python/Zala_task/Zala_task/data/output_labels",
        window_title="YOLO Аннотации"
    )
    app.run()
















