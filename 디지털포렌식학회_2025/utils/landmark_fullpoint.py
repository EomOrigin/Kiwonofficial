import cv2
import mediapipe as mp
import numpy as np
from utils.landmark_indices import EYE_IDXS, NOSE_IDXS, MOUTH_IDXS

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3
)

def extract_landmark_full_coords(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[❌] 이미지 로드 실패: {image_path}")
        return []

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(img_rgb)

    if not res.multi_face_landmarks:
        print(f"[❌] 얼굴 탐지 실패: {image_path}")
        return []

    lm = res.multi_face_landmarks[0].landmark
    idxs = list(EYE_IDXS) + list(NOSE_IDXS) + list(MOUTH_IDXS)

    coords = []
    for idx in idxs:
        if idx >= len(lm):
            print(f"[⚠️] 유효하지 않은 인덱스 {idx} in {image_path}")
            continue
        p = lm[idx]
        coords.append((int(p.x * w), int(p.y * h)))

    if len(coords) != 58:
        print(f"[⚠️] landmark 수 부족 ({len(coords)})개 → {image_path}")

    return coords
