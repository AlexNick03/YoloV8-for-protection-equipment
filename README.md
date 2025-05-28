# 📂 YOLOv8 for Protection Equipment Detection

This project utilizes a YOLOv8 model trained to detect protective equipment in images and in real-time using a webcam.

---

## 📦 Training Data
The images used for training were obtained from [Roboflow](https://roboflow.com/) under its terms and conditions.

⚠️ **Note:** The training images are **not included** in this repository and have **not been uploaded**.  
Each user is responsible for creating their own dataset and generating a local `data.yaml` file.

---

## 🔍 Testing
For testing, please use the provided `real_time.py` script, which will perform real-time detection using the webcam and the trained YOLOv8 model (`best.pt`).

---

## ⚙️ Setup and Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/AlexNick03/YoloV8-for-protection-equipment.git
cd YoloV8-for-protection-equipment
