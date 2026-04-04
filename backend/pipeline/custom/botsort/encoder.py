# Lightweight appearance encoder for BOTSORT ReID.
#
# Extracts HSV color histogram features from detection crops.
# Similar to NvDCF's ColorNames approach — uses color distribution
# as an appearance signature for cosine-distance matching.
# No torch dependency — pure OpenCV + numpy.

import cv2
import numpy as np

# Histogram bins: 16 hue × 4 sat × 4 val = 256-dim feature vector
_H_BINS = 16
_S_BINS = 4
_V_BINS = 4
_FEAT_DIM = _H_BINS + _S_BINS + _V_BINS  # 24-dim (concatenated, not joint)


def _extract_crop(img: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    """Extract and resize a crop from the image."""
    h, w = img.shape[:2]
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(w, int(xyxy[2]))
    y2 = min(h, int(xyxy[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _histogram_feature(crop: np.ndarray) -> np.ndarray:
    """Compute normalized HSV histogram feature vector from a BGR crop."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Separate channel histograms (more robust than joint histogram)
    h_hist = cv2.calcHist([hsv], [0], None, [_H_BINS], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [_S_BINS], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [_V_BINS], [0, 256]).flatten()

    feat = np.concatenate([h_hist, s_hist, v_hist])

    # L2 normalize for cosine distance
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat /= norm
    return feat.astype(np.float32)


class HistogramEncoder:
    """Lightweight ReID encoder using HSV color histograms.

    Compatible with BOTSORT's encoder interface:
        encoder(img, bboxes) -> list[np.ndarray]

    Each feature is a 24-dim L2-normalized vector suitable for
    cosine distance in BOTSORT's embedding_distance().
    """

    def __call__(
        self, img: np.ndarray, bboxes: np.ndarray,
    ) -> list[np.ndarray]:
        """Extract appearance features for each detection.

        Args:
            img: BGR frame (H, W, 3).
            bboxes: (N, 4) array of [x1, y1, x2, y2] bounding boxes.

        Returns:
            List of N feature vectors (24-dim each).
        """
        features = []
        for bbox in bboxes:
            crop = _extract_crop(img, bbox)
            if crop is not None:
                features.append(_histogram_feature(crop))
            else:
                features.append(np.zeros(_FEAT_DIM, dtype=np.float32))
        return features
