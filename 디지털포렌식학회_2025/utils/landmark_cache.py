# utils/landmark_cache.py

import os
import pickle

def load_landmark_cache(cache_file="landmark_cache.pkl"):
    """
    지정된 cache_file에서 landmark 캐시를 불러옵니다.
    cache_file이 없으면 빈 dict를 반환합니다.
    """
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded landmark cache with {len(cache)} entries from {cache_file}")
        return cache
    else:
        print(f"No cache found at {cache_file}, starting fresh.")
        return {}

def save_landmark_cache(cache, cache_file="landmark_cache.pkl"):
    """
    cache 딕셔너리를 지정된 cache_file에 저장합니다.
    파일이 없으면 새로 생성하고, 있으면 덮어씁니다.
    """
    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved landmark cache with {len(cache)} entries to {cache_file}")

def cached_extract_landmark_coords(image_path, extract_func, cache):
    """
    cache에 image_path가 있으면 해당 값을 반환하고,
    없으면 extract_func로 랜드마크를 추출해 cache에 저장한 뒤 반환합니다.
    """
    if image_path in cache:
        return cache[image_path]
    else:
        lm = extract_func(image_path)
        cache[image_path] = lm
        return lm
