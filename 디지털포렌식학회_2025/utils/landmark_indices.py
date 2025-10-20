# utils/face_landmark_indices.py

# 왼쪽 눈 영역 (FACEMESH_LEFT_EYE)
LEFT_EYE_IDXS = {
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246
}
# 오른쪽 눈 영역 (FACEMESH_RIGHT_EYE)
RIGHT_EYE_IDXS = {
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
}
# 둘 합치기
EYE_IDXS = LEFT_EYE_IDXS | RIGHT_EYE_IDXS

# 코 전체: nose tip + nostrils + mid-nose
NOSE_IDXS = {1, 2, 98, 327, 168, 6}

# 입술 전체 (FACEMESH_LIPS)
MOUTH_IDXS = {
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
}
