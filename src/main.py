import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import utils 

VIDEO_PATH = 'videos/test.mp4'
OUTPUT_PATH = 'result/final_output.mp4'
MODEL_PATH = 'runs/football_result_high_res/weights/best.pt'

def main():
    print(f"Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(">>> Step 1: Processing Ball Trajectory (Interpolation)...")
    final_ball_pos = utils.get_interpolated_ball_positions(model, VIDEO_PATH, total_frames)
    print("Step 1 Complete! Ball trajectory interpolated.")

    print(">>> Step 2: Running Tracking & Rendering...")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    track_history = defaultdict(lambda: [])
    class_history = defaultdict(lambda: [])
    SMOOTH_FACTOR = 3

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model.track(frame, persist=True, imgsz=1280, conf=0.25, 
                              tracker="bytetrack.yaml", verbose=False, 
                              classes=[utils.CLS_PLAYER, utils.CLS_REFEREE])

        if frame_idx in final_ball_pos:
            pos = final_ball_pos[frame_idx]
            utils.draw_ball_marker(frame, pos['x'], pos['y'])

        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist()
            cls_ids = boxes.cls.int().cpu().tolist()
            coords = boxes.xyxy.int().cpu().tolist()

            for box, track_id, cls_id in zip(coords, track_ids, cls_ids):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, y2

                class_history[track_id].append(cls_id)
                if len(class_history[track_id]) > 30: 
                    class_history[track_id].pop(0)
                
                final_cls = utils.resolve_class_voting(class_history[track_id])

                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > SMOOTH_FACTOR:
                    track_history[track_id].pop(0)
                
                smooth_x = int(np.mean([p[0] for p in track_history[track_id]]))
                smooth_y = int(np.mean([p[1] for p in track_history[track_id]]))

                utils.draw_person_marker(frame, smooth_x, smooth_y, track_id, final_cls)

        out.write(frame)
        
        if frame_idx % 100 == 0:
            print(f"  Rendering frame {frame_idx}/{total_frames}...")
        frame_idx += 1

    cap.release()
    out.release()
    print(f"All Done! Result saved at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
