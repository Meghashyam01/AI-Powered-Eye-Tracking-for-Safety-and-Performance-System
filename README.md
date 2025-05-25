# 👁️ AI-Powered Eye Tracking for Safety and Performance System

A real-time AI-based system designed to monitor eye movements and blinking patterns to detect drowsiness, inattention, or fatigue, enhancing safety and improving performance in critical environments like driving, industrial operations, and workplace monitoring.

---

## 🚀 Project Overview

This project utilizes computer vision and machine learning techniques to:

- Track eye movement using a webcam or external camera.
- Detect drowsiness based on blink rate and eye aspect ratio (EAR).
- Alert the user in real-time through audio or visual notifications.
- Provide logs for performance monitoring and safety analytics.

---

## 🧠 Technologies Used

- **Python**
- **OpenCV** – for real-time video capture and image processing.
- **Dlib** – for facial landmark detection.
- **scikit-learn / TensorFlow** – for training machine learning models (optional).
- **NumPy, imutils** – for numerical and image utility operations.
- **Pyttsx3 / Playsound / Pygame** – for voice alerts or alarm sounds.

---

## 📁 Features

- 🔍 **Eye Detection & Tracking**
- 😴 **Drowsiness & Fatigue Detection**
- 🧠 **Cognitive Load Monitoring** *(optional enhancement)*
- 📊 **Performance Analytics**
- 🔊 **Real-Time Alerts**
- 📦 **Modular Code Design**

---

## ⚙️ How It Works

1. Captures real-time video from webcam.
2. Detects facial landmarks using Dlib's 68-point model.
3. Calculates Eye Aspect Ratio (EAR) to monitor eye closure.
4. If EAR is below a threshold for a specific duration, triggers an alert.
5. Logs activity and alert history for later analysis.

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/AI-Powered-Eye-Tracking-for-Safety-and-Performance-System.git
cd AI-Powered-Eye-Tracking-for-Safety-and-Performance-System
pip install -r requirements.txt

## ✍️ Author

**Sana Ali** – AI Intern | ML Enthusiast | Safety AI Innovator

---

## 🛡️ License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙌 Acknowledgments

- Dlib Facial Landmark Detection by [Davis King](http://dlib.net/)
- OpenCV community
- PyImageSearch tutorials and resources



