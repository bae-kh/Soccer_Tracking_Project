# ⚽ Robust Soccer Video Analysis System
> **Object Tracking & Trajectory Interpolation for Sports Analytics**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)

## 📌 Introduction
This project implements a robust multi-object tracking system for soccer videos. It addresses common challenges in sports analytics, such as **motion blur** and **occlusion**, by integrating deep learning with classical engineering algorithms.

## 🎥 Demo
![Demo GIF](./assets/demo.gif)
*(여기에 아까 만든 결과물 움짤을 넣으세요. GIF 파일이 assets 폴더에 있어야 합니다)*

## 🚀 Key Features

### 1. Trajectory Reconstruction (Interpolation)
- **Problem:** Fast-moving balls often disappear due to motion blur (False Negative).
- **Solution:** Utilized **Pandas Linear Interpolation** to mathematically recover missing coordinates in the ball's trajectory.

### 2. Stable Classification (Majority Voting)
- **Problem:** Player IDs flicker between 'Player' and 'Referee' during occlusion.
- **Solution:** Implemented a **Temporal Majority Voting** algorithm using a Queue (Window size=30) to stabilize class prediction.

### 3. Advanced Tracking
- Integrated **ByteTrack** to handle low-confidence detections and maintain ID consistency.

## 🛠️ Installation & Usage
1. Clone the repository
   ```bash
   git clone [https://github.com/your-username/Soccer-Video-Analysis.git](https://github.com/your-username/Soccer-Video-Analysis.git)
   ```
2. Install dependencies
   ```bash
    pip install -r requirements.txt
   ```
3. Run the code
   ```bash
   python src/main.py --source video.mp4
   ```
