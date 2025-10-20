import cv2
import mediapipe as mp
import numpy as np

def weighted_average_points(landmarks, indices, weights=None, w=224, h=224):
    pts = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        pts.append([lm.x * w, lm.y * h])
    pts = np.array(pts)
    if weights is not None:
        weights = np.array(weights)
        avg_x = np.average(pts[:, 0], weights=weights)
        avg_y = np.average(pts[:, 1], weights=weights)
    else:
        avg_x = np.mean(pts[:, 0])
        avg_y = np.mean(pts[:, 1])
    return (int(avg_x), int(avg_y))

def extract_landmark_5pt_coords(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not loaded properly")
        img = cv2.resize(img, (224, 224))
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            results = face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                raise ValueError("No face detected in the image")
            face_landmarks = results.multi_face_landmarks[0]

            left_eye_indices    = [33, 133, 159, 145]
            right_eye_indices   = [263, 362, 386, 374]
            nose_indices        = [1, 2, 168]
            mouth_left_indices  = [61, 146, 91]
            mouth_right_indices = [291, 375, 321]

            left_eye_center = weighted_average_points(face_landmarks, left_eye_indices, weights=[1.0, 1.2, 1.2, 1.0], w=w, h=h)
            right_eye_center = weighted_average_points(face_landmarks, right_eye_indices, weights=[1.0, 1.2, 1.2, 1.0], w=w, h=h)
            nose_center = weighted_average_points(face_landmarks, nose_indices, w=w, h=h)
            mouth_left_center = weighted_average_points(face_landmarks, mouth_left_indices, w=w, h=h)
            mouth_right_center = weighted_average_points(face_landmarks, mouth_right_indices, w=w, h=h)

            return [left_eye_center, right_eye_center, nose_center, mouth_left_center, mouth_right_center]

    except Exception as e:
        print(f"[Warning] Landmark extraction failed for {image_path}: {e}")
        return []
