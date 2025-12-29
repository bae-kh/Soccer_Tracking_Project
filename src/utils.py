import cv2
import numpy as np
import pandas as pd
from collections import Counter

CLS_BALL = 0
CLS_PLAYER = 2
CLS_REFEREE = 3

COLOR_PLAYER = (0, 0, 255)
COLOR_REFEREE = (0, 255, 255)
COLOR_BALL = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)

def get_interpolated_ball_positions(model, video_path, total_frames):
    print(f"Checking ball trajectory from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    ball_positions = {}
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, imgsz=1280, conf=0.1, verbose=False)
        
        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id == CLS_BALL:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = y1 - 10
                    ball_positions[frame_idx] = (center_x, center_y)
                    break
        
        if frame_idx % 500 == 0:
            print(f"  Scanning frame {frame_idx}/{total_frames}...")
        frame_idx += 1
    
    cap.release()

    df = pd.DataFrame.from_dict(ball_positions, orient='index', columns=['x', 'y'])
    df = df.reindex(range(total_frames))
    df = df.interpolate(method='linear', limit=30, limit_direction='both')
    
    return df.to_dict('index')

def resolve_class_voting(history_queue):
    if not history_queue:
        return None
    return Counter(history_queue).most_common(1)[0][0]

def draw_ball_marker(frame, x, y):
    if np.isnan(x) or np.isnan(y):
        return
    
    bx, by = int(x), int(y)
    pts = np.array([
        [bx, by], 
        [bx-10, by-20], 
        [bx+10, by-20]
    ])
    cv2.drawContours(frame, [pts], 0, COLOR_BALL, -1)

def draw_person_marker(frame, x, y, track_id, cls_id):
    if cls_id == CLS_PLAYER:
        color = COLOR_PLAYER
    elif cls_id == CLS_REFEREE:
        color = COLOR_REFEREE
    else:
        return

    cv2.ellipse(frame, (x, y), (35, 12), 0, 0, 360, color, 2)
    
    id_text = str(track_id)
    (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    cv2.rectangle(frame, (x - tw//2 - 5, y + 10), (x + tw//2 + 5, y + 10 + th + 10), color, -1)
    cv2.putText(frame, id_text, (x - tw//2, y + 25), cv2.FONT_HERSHEY_SIMPLEX,
