
"""  
Автор: Лейман М.А.   
Дата создания: 22.03.2025 

детектирование объектов на видео
"""

from ultralytics import YOLO
import cv2
import os



def video_detection(path_current, model, video_path, output_video_path, target_classes=None):
    model_path = os.path.join(path_current, model)
    print("model_path", model_path)
    model = YOLO(model_path)

    screen_width = 1000
    screen_height = 1000
    conf = 0.1
    iou = 0.4
    video_path = os.path.join(path_current, video_path)
    output_video_path = os.path.join(path_current, output_video_path)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (screen_width, screen_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (screen_width, screen_height))
        results = model.track(frame_resized, persist=True, conf=conf, iou=iou, agnostic_nms=True, verbose=False)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None] * len(boxes)

            for box, label, score in zip(boxes, labels, scores):
                class_name = model.names[int(label)]
                
                # Фильтрация по нужным классам
                if target_classes and class_name not in target_classes:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame_resized, f'{class_name} {score:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out.write(frame_resized)
        cv2.imshow('Frame', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

target_classes=['building', 'plane', 'fish net', 'landslide', 'pool', 'house', 'quarry', 'ship', 'vehicle', 'well', 'harbor', 'big vehicle', 'helicopter']



if __name__ == '__main__':
    # Получаем текущую рабочую папку  
    current_directory = os.getcwd()  
    model = 'Zala_task/Zala_task/'
    path_current = os.path.join(current_directory, model)
    
    print("current_directory", path_current)
    # Параметры
    video_path = "video/video3.avi"  # Путь к входному видео

    s_model_path_UAVOD_6 = "models/11s_best_UAVOD6.pt"
    s_model_path_UAVOD_5 = "models/11s_best_UAVOD5.pt"  #  Модель обученная на датасете UAVOD + DOTA
    s_model_path_UAVOD_4 = "models/11s_best_UAVOD4.pt"  #  Модель обученная на датасете UAVOD
    s_model_path_UAVOD_3 = "models/11s_best_UAVOD3.pt"
 

    n_output_video_path = "video/video_yolo_n.avi"  

    n_model_path_UAVOD_2 = "12n_best_UAVOD2.pt" 
    n_model_path_UAVOD_3 = "12n_best_UAVOD3.pt" 
 


    video_detection(path_current, s_model_path_UAVOD_4, video_path = "videos/video_in/video_12.avi", output_video_path = "videos/video_out/video_yolo_s_UAVOD_231.avi", target_classes=target_classes)







