🎭 Real-Time Emotion Detection & AI Mood Support System
An intelligent, interactive Streamlit application that understands human emotion and responds with AI-driven support.

This project is live at : 

🌟 Overview

This project is a full-stack Emotion AI system that analyzes human emotions from both images and webcam snapshots, and then responds intelligently using:

🎬 GIF reactions

🎵 Emotion-matched music

🧠 AI-generated motivational messages

🟩 Face bounding boxes

📈 Analytics dashboards

🧾 Emotion history tracking

Built using DeepFace, OpenCV, TensorFlow, HuggingFace, and Streamlit, this application demonstrates strong skills across:

Machine Learning

Computer Vision

Real-Time Inference

Full-stack AI Integration

Human-Computer Interaction

Data Visualization

This project is ideal for mental wellness, smart classrooms, user engagement systems, and AI-powered assistants — and is designed to stand out on a resume.

🚀 Features
🎥 1. Dual Input Modes

Webcam Mode → Capture a frame and analyze emotion instantly

Image Mode → Upload any photo for emotion detection

😊 2. Advanced DeepFace Emotion Recognition

Multi-face detection

Bounding boxes + labels

Dominant emotion + probability scores

Works with noisy/low-light images

Runs fully locally except for AI message generation

🎬 3. Automatic GIF Reaction System

Every emotion triggers a curated GIF, creating a natural, human-like response.

🎵 4. Mood-Based Music Recommendations

Music playlists selected to match emotional state:

Emotion	Music
Happy	Celebration Vibes
Sad	Calm Piano
Angry	Relaxation Music
Fear	Soothing Ambience
Neutral	Lofi Chill
Surprise	Fun Pop
Disgust	Calming Tracks

Embedded YouTube player ensures instant playback.

💬 5. AI-generated Emotional Support

Using HuggingFace Gemma-2B, the app generates personalized, motivational messages such as:

“It’s okay to feel overwhelmed — remember you're stronger than your worries.”

Offline fallback messages ensure reliability.

📊 6. Emotion Analytics Dashboard

Includes:

Bar chart → Emotion frequency

Line chart → Emotion trend over time

Probability distribution graph

Exportable CSV logs

Perfect for reporting, research, or user behavior tracking.

🧹 7. Clean, Compact, Professional UI

Two-column layout

Compact images (no oversized visuals)

Easy scrolling & readability

Works on laptops + Streamlit Cloud flawlessly

🧠 Tech Stack
Layer	Technologies Used
UI / Frontend	Streamlit
Backend ML	DeepFace (CNN-based), TensorFlow
Image Processing	OpenCV, PIL
AI Text Generation	HuggingFace Inference API (Gemma-2B)
Data Analytics	Pandas, NumPy, Matplotlib
Deployment	Streamlit Cloud, GitHub
📂 Project Structure
Emotion-AI/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies for deployment
├── README.md             # Documentation
└── assets/               # (Optional) GIFs or icons

🛠️ Installation & Local Usage
1️⃣ Clone Repository
git clone https://github.com/<your-username>/Emotion-AI.git
cd Emotion-AI

2️⃣ Create Virtual Environment
python -m venv emotion_env

3️⃣ Activate the Environment

Windows

emotion_env\Scripts\activate

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Run Application
streamlit run app.py

🌐 Deploy on Streamlit Cloud

Upload the project to your GitHub

Go to https://share.streamlit.io

Select your repository

Choose app.py

Deploy

Your app becomes live instantly and accessible anywhere.

🔍 Emotion Analytics (Examples)
🟦 Emotion Frequency Chart

Shows which emotions occur most often.

📉 Emotion Trend Line

Plots how emotion changes over time — ideal for behavioral analysis.

🎯 Probability Distribution

Displays confidence levels for each emotion from the last detection.

📥 CSV Export

Download complete emotion history for further analysis.

🧑‍💻 Author

Vaishnavi Newalkar


📜 License

Open-source under MIT License.
