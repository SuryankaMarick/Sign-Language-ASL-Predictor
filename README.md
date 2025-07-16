# 🤟 ASL Sign Language Alphabet Predictor (Beginner Project)

This is my first Python-based project — a simple Sign Language (ASL) Alphabet Predictor using a webcam. It uses **computer vision** and **machine learning** to recognize American Sign Language (ASL) alphabets from hand gestures.

🔍 Currently, the model achieves around **30% accuracy**, but it's a solid starting point, and I’m actively learning and improving it!

---

## 🛠️ Tools & Technologies Used

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![cvzone](https://img.shields.io/badge/CVZone-00BFFF?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white)

---

## 🎯 Project Features

- Real-time webcam-based prediction
- Predicts ASL Alphabets (A-Z)
- Hand tracking using `cvzone` + `OpenCV`
- Trained on custom or publicly available ASL image dataset
- Simple interface with live prediction output

---

## 📊 Model Performance

- 🔹 Current Accuracy: ~30%
- 🔹 Model trained for learning purposes (not production-grade yet)
- 🔹 Dataset used: ASL alphabet hand signs (static images)

---

## 🚀 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/SuryankaMarick/Sign-Language-ASL-Predictor.git
   cd Sign-Language-ASL-Predictor
   pip install -r requirements.txt
   python3 test.py
