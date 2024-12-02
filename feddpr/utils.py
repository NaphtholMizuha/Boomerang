import numpy as np

def cos_sim(x: np.ndarray, y: np.ndarray) -> float:
    return y.dot(x) / (np.linalg.norm(x) * np.linalg.norm(y, axis=1))

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.sum(x)

def z_score_outliers(x: np.ndarray, thr=3):
    z_scores = (x - x.mean()) / x.std()
    return np.abs(z_scores) > thr