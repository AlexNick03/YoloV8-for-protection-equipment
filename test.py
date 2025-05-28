import os
from ultralytics import YOLO
from dotenv import load_dotenv

    # Calea către modelul YOLO antrenat
MODEL_PATH = os.getenv("MODEL_PATH")

    # Încarcă modelul YOLO
model = YOLO(MODEL_PATH)
    
    # Folderul cu imaginile de testare
IMAGE_FOLDER = os.getenv("IMAGE_FOLDER")  # Modifică dacă imaginile sunt în alt folder

    
    # Folderul unde vor fi salvate rezultatele
OUTPUT_FOLDER = "predictions"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Detectare pe fiecare imagine din folder
for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(IMAGE_FOLDER, filename)
        print(f"Processing: {filename}")

        results = model.predict(
                source=img_path,
                conf=0.25,  # Confidența minimă pentru detectări
                save=True,  # Salvează imaginile cu detectări
                project=OUTPUT_FOLDER,  # Salvează în folderul specificat
                name="results"  # Subfolder specific în OUTPUT_FOLDER
            )

            # Afișează rezultatele în terminal
        for r in results:
            if hasattr(r, "boxes") and r.boxes is not None:
                for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = "free" if cls_id == 0 else "occupied"
                        print(f"Detected {label} spot at {box.xyxy[0].tolist()}")
            else:
                    print("No detections found.")
    
print("Inference complete! Results saved in:", OUTPUT_FOLDER)

