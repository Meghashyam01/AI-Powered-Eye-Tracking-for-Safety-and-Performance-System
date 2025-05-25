# AI-Powered-Eye-Tracking-for-Safety-and-Performance-System

A real-time AI-based system designed to monitor eye movements and blinking patterns to detect drowsiness, inattention, or fatigue, enhancing safety and improving performance in critical environments like driving, industrial operations, and workplace monitoring.

🚀 Project Overview
This project utilizes computer vision and machine learning techniques to:

Track eye movement using a webcam or external camera.

Detect drowsiness based on blink rate and eye aspect ratio (EAR).

Alert the user in real-time through audio or visual notifications.

Provide logs for performance monitoring and safety analytics.

🧠 Technologies Used
Python

OpenCV – for real-time video capture and image processing.

Dlib – for facial landmark detection.

scikit-learn / TensorFlow – for training machine learning models (optional for advanced features).

NumPy, imutils – for numerical and image utility operations.

Pyttsx3 / Playsound / Pygame – for voice alerts or alarm sounds.

📁 Features
🔍 Eye Detection & Tracking

😴 Drowsiness & Fatigue Detection

🧠 Cognitive Load Monitoring (optional enhancement)

📊 Performance Analytics (log output for post-session analysis)

🔊 Real-Time Alerts (audio/visual)

📦 Modular Code Design for easy extension

⚙️ How It Works
Captures real-time video from webcam.

Detects facial landmarks using Dlib's 68-point model.

Calculates Eye Aspect Ratio (EAR) to monitor eye closure.

If EAR is below a threshold for a specific duration, triggers an alert.

Logs activity and alert history for later analysis.

🛠️ Installation
bash
Copy
Edit
git clone https://github.com/yourusername/AI-Powered-Eye-Tracking-for-Safety-and-Performance-System.git
cd AI-Powered-Eye-Tracking-for-Safety-and-Performance-System
pip install -r requirements.txt
▶️ Usage
bash
Copy
Edit
python eye_tracking_system.py
Make sure your webcam is connected and accessible.

📈 Future Enhancements
Integration with deep learning models for more robust predictions.

Add head pose estimation to detect distraction.

Integration with dashboard analytics using Streamlit or Dash.

Deploy on Raspberry Pi or edge devices.

Support for thermal camera input (for industrial safety environments).

📄 Folder Structure
bash
Copy
Edit
AI-Eye-Tracking/
│
├── eye_tracking_system.py        # Main script
├── utils/
│   └── eye_aspect_ratio.py       # Helper function for EAR
├── models/                       # (Optional ML/DL models)
├── requirements.txt
├── README.md
└── alert_sounds/
    └── alarm.wav
✍️ Authors
Sana Ali – AI Intern | Machine Learning Enthusiast | Safety AI Innovator

🛡️ License
This project is licensed under the MIT License. See the LICENSE file for details.

🙌 Acknowledgments
Dlib Facial Landmark Detection by Davis King

OpenCV community for powerful real-time image processing tools

Tutorials from PyImageSearch and other open-source contributors
