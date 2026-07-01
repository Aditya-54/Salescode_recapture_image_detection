"""Spot the Fake Photo -- REAL vs PHOTO-OF-A-SCREEN detector.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)

Approach
--------
No deep neural network. A small set of classical image-forensics features
(frequency-domain moire/periodicity signal, gradient-orientation peakiness,
sharpness, edge density, LBP texture, color statistics, highlight/glare
stats, mild blockiness) feeds a tiny soft-voting ensemble of Logistic
Regression + shallow Gradient Boosting (~128 KB model file, trained in
train.py / training_notebook.ipynb).

The single strongest signal: a photo of a screen has the screen's own pixel
grid beating against the camera sensor's grid, producing sharp, localized
peaks in the FFT magnitude spectrum once the smooth radial falloff shared by
all natural photos is subtracted out (see fft_moire_features below).

See NOTE.md for methodology, honest accuracy numbers, and latency/cost.
"""

import sys
from pathlib import Path

import numpy as np
import cv2
import joblib
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

MODEL_PATH = Path(__file__).parent / "screen_detector_model.joblib"
IMG_SIZE = 512

_MODEL_CACHE = None  # loaded once, reused across calls in the same process


def skew(x):
    """Fisher-Pearson skewness, implemented in plain numpy instead of
    scipy.stats (scipy.stats import alone costs ~700ms of cold-start
    latency for this one statistic -- not worth it)."""
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - x.mean()) / s) ** 3))


# ------------------------------------------------------------------ #
# Image loading (fast decode for large phone photos)
# ------------------------------------------------------------------ #
def load_image(path, size=IMG_SIZE):
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)
    if img is None or max(img.shape[:2]) < size:
        img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = size / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# ------------------------------------------------------------------ #
# Feature extraction
# ------------------------------------------------------------------ #
def laplacian_variance(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def fft_moire_features(gray):
    g = gray.astype(np.float32)
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    g = g * win

    f = np.fft.fft2(g)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_mag = np.log1p(magnitude)

    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.int32)
    max_r = dist.max()

    radial_mean = np.bincount(dist.ravel(), log_mag.ravel()) / np.maximum(np.bincount(dist.ravel()), 1)
    radial_profile = radial_mean[dist]
    residual = log_mag - radial_profile

    mask_mid_high = dist > (0.05 * max_r)
    resid_mid_high = residual[mask_mid_high]
    peak_score = resid_mid_high.max() if resid_mid_high.size else 0.0
    peak_energy = np.mean(resid_mid_high > 3.0) if resid_mid_high.size else 0.0
    resid_std = resid_mid_high.std() if resid_mid_high.size else 0.0

    low_mask = dist < (0.08 * max_r)
    high_mask = dist > (0.25 * max_r)
    low_energy = magnitude[low_mask].mean()
    high_energy = magnitude[high_mask].mean()
    ratio = high_energy / (low_energy + 1e-8)

    return [magnitude.mean(), magnitude.std(), low_energy, high_energy, ratio,
            peak_score, peak_energy, resid_std]


def gradient_features(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = (np.arctan2(gy, gx) * 180 / np.pi) % 180

    strong = magnitude > (magnitude.mean() + magnitude.std())
    if strong.sum() > 10:
        hist, _ = np.histogram(angle[strong], bins=18, range=(0, 180), weights=magnitude[strong])
        hist = hist / (hist.sum() + 1e-8)
        orientation_peakiness = hist.max()
        axis_bins = np.concatenate([hist[0:2], hist[8:10], hist[16:18]])
        axis_energy = axis_bins.sum()
    else:
        orientation_peakiness = 0.0
        axis_energy = 0.0

    return [magnitude.mean(), magnitude.std(), magnitude.max(), orientation_peakiness, axis_energy]


def edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size


def lbp_histogram(gray):
    radius, points = 2, 16
    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), density=True)
    return hist


def color_features(image):
    features = []
    for c in range(3):
        data = image[:, :, c].ravel().astype(np.float32)
        features.extend([data.mean(), data.std(), skew(data)])
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    features.append(hsv[:, :, 1].mean())
    features.append(hsv[:, :, 1].std())
    features.append(hsv[:, :, 2].mean())
    return features


def highlight_features(gray):
    overexposed = np.mean(gray > 245)
    underexposed = np.mean(gray < 10)
    p99 = np.percentile(gray, 99)
    dynamic_range = float(gray.max()) - float(gray.min())
    return [overexposed, underexposed, p99, dynamic_range]


def blockiness(gray, block=8):
    g = gray.astype(np.float32)
    h, w = g.shape
    h = h - h % block
    w = w - w % block
    g = g[:h, :w]
    dx = np.abs(np.diff(g, axis=1))
    boundary_cols = np.arange(block - 1, w - 1, block)
    if len(boundary_cols) == 0:
        return [0.0]
    boundary_energy = dx[:, boundary_cols].mean()
    overall_energy = dx.mean() + 1e-8
    return [boundary_energy / overall_energy]


def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    feats = []
    feats.append(laplacian_variance(gray))
    feats.extend(fft_moire_features(gray))
    feats.extend(gradient_features(gray))
    feats.append(shannon_entropy(gray))
    feats.append(edge_density(gray))
    feats.extend(color_features(image))
    feats.extend(highlight_features(gray))
    feats.extend(blockiness(gray))
    feats.extend(lbp_histogram(gray))
    return np.array(feats, dtype=np.float64).reshape(1, -1)


def _get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = joblib.load(MODEL_PATH)["model"]
    return _MODEL_CACHE


# ------------------------------------------------------------------ #
# The interface
# ------------------------------------------------------------------ #
def predict(image_path: str) -> float:
    model = _get_model()
    img = load_image(image_path)
    feats = extract_features(img)
    prob_screen = model.predict_proba(feats)[0, 1]
    return float(prob_screen)


if __name__ == "__main__":
    print(predict(sys.argv[1]))