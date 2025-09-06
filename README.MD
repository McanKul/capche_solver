# Captcha Solver 🔎🤖

Python + Playwright + TensorFlow/Keras based framework for **collecting, labeling, training and solving captchas**.  
Goal: fetch captcha images from websites, manually label them, fine-tune a model, and then solve captchas automatically.

⚠️ **Disclaimer:** This project is for **educational, research, and personal use only**. Do not use it to violate third-party websites’ terms of service.

---

## ✨ Features

- [x] **Captcha collection** with Playwright (`Captcha.aspx` or similar endpoints)
- [x] Simple **manual labeling GUI** (`labeler.py`)
- [x] **Model training / fine-tuning** with Keras (`finetune.py`)
- [x] Inference pipeline: fetch → preprocess → predict
- [ ] (Planned) CTC support for variable-length captchas
- [ ] (Planned) Active learning: prioritize labeling of misclassified samples

---

## 📂 Project Structure

captcha-solver/
│
├── collect_captchas.py # Collect captcha images (Playwright)
├── labeler.py # Tkinter labeling tool
├── finetune.py # Fine-tune Keras model
├── solver.py # Inference pipeline
│
├── dataset/
│ ├── raw/ # Raw collected captchas
│ ├── labeled/ # Labeled images (__LABEL.png format)
│ └── labeled/labels.csv
│
├── models/
│ ├── my_model.keras # Base model (pretrained)
│ └── ft_model.keras # Fine-tuned model
│
└── README.md


---

## ⚙️ Installation

Recommended Python: **3.11+**

```bash
git clone https://github.com/<your-username>/captcha-solver.git
cd captcha-solver

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium
