import cv2
import mediapipe as mp
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

MOUTH_LANDMARKS = list(range(61, 88))

def extract_mouth(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(save_dir, exist_ok=True)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            xs = [int(landmarks[i].x * w) for i in MOUTH_LANDMARKS]
            ys = [int(landmarks[i].y * h) for i in MOUTH_LANDMARKS]

            x1, x2 = max(min(xs) - 5, 0), min(max(xs) + 5, w)
            y1, y2 = max(min(ys) - 5, 0), min(max(ys) + 5, h)

            mouth = frame[y1:y2, x1:x2]
            mouth = cv2.resize(mouth, (64, 64))

            cv2.imwrite(os.path.join(save_dir, f"{frame_id}.jpg"), mouth)
            frame_id += 1

    cap.release()
    print(f"[OK] Saved {frame_id} mouth frames.")
