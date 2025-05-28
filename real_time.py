from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import sys
import os
from datetime import datetime  # Pentru a genera numele fișierului

# Funcție pentru a obține calea corectă către resurse
def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Încarcă modelul YOLO
model_path = get_resource_path("best.pt")
model = YOLO(model_path)

# Variabile globale pentru controlul camerei și salvarea video
running = False
cap = None
video_writer = None
output_folder = "records"  # Folderul unde vor fi salvate videoclipurile

# Creează folderul "records" dacă nu există
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Funcție pentru detectarea pe imagini
def detect_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    image = cv2.imread(file_path)
    results = model.predict(source=image, conf=0.4, imgsz=640, device=0)
    annotated_image = results[0].plot()

    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    annotated_image = Image.fromarray(annotated_image)
    annotated_image = ImageTk.PhotoImage(annotated_image)

    image_label.config(image=annotated_image)
    image_label.image = annotated_image

# Funcție pentru detectarea cu camera web
def detect_camera():
    global running, cap, video_writer

    # Deschide camera web
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Eroare: Camera nu poate fi deschisă.")
        return

    # Pregătește VideoWriter pentru a salva videoclipul
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Generează numele fișierului folosind data și ora curentă
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(output_folder, f"recording_{current_time}.mp4")

    # Inițializează VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pentru formatul MP4
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    running = True

    # Funcție pentru actualizarea frame-urilor
    def update_frame():
        if running:
            ret, frame = cap.read()
            if ret:
                # Realizează detectarea pe cadrul curent
                results = model.predict(source=frame, conf=0.4, imgsz=640, device=0)
                annotated_frame = results[0].plot()

                # Scrie cadrul în fișierul video
                video_writer.write(annotated_frame)

                # Convertim frame-ul pentru afișare în Tkinter
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                annotated_frame = Image.fromarray(annotated_frame)
                annotated_frame = ImageTk.PhotoImage(annotated_frame)

                # Actualizează imaginea în interfață
                image_label.config(image=annotated_frame)
                image_label.image = annotated_frame

            # Repetă funcția la fiecare 10 ms
            root.after(10, update_frame)

    # Pornim actualizarea frame-urilor
    update_frame()

# Funcție pentru oprirea camerei
def stop_camera():
    global running, cap, video_writer
    if running:
        running = False  # Oprește bucla de actualizare
    if cap is not None:
        cap.release()  # Eliberează camera
        cap = None
    if video_writer is not None:
        video_writer.release()  # Închide VideoWriter
        video_writer = None
    image_label.config(image=None)  # Șterge imaginea afișată

# Creăm interfața grafică
root = tk.Tk()
root.title("YOLOv8 Detectare Obiecte")

# Buton pentru încărcarea imaginilor
btn_image = tk.Button(root, text="Încarcă și detectează imagine", command=detect_image)
btn_image.pack(pady=10)

# Buton pentru pornirea camerei
btn_camera = tk.Button(root, text="Pornește camera", command=detect_camera)
btn_camera.pack(pady=10)

# Buton pentru oprirea camerei
btn_stop = tk.Button(root, text="Oprește camera", command=stop_camera)
btn_stop.pack(pady=10)

# Etichetă pentru afișarea imaginilor
image_label = tk.Label(root)
image_label.pack()

# Pornim bucla principală a interfeței
root.mainloop()