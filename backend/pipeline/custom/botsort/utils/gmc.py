# Vendored from ultralytics v8.4 (AGPL-3.0)
from __future__ import annotations

import copy
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class GMC:
    """Generalized Motion Compensation using ORB, SIFT, ECC, or Sparse Optical Flow."""

    def __init__(self, method: str = "sparseOptFlow", downscale: int = 2) -> None:
        super().__init__()
        self.method = method
        self.downscale = max(1, downscale)

        if self.method == "orb":
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif self.method == "sift":
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif self.method == "ecc":
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        elif self.method == "sparseOptFlow":
            self.feature_params = dict(
                maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04
            )
        elif self.method in {"none", "None", None}:
            self.method = None
        else:
            raise ValueError(f"Unknown GMC method: {method}")

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

    def apply(self, raw_frame: np.ndarray, detections: list | None = None) -> np.ndarray:
        if self.method in {"orb", "sift"}:
            return self._apply_features(raw_frame, detections)
        elif self.method == "ecc":
            return self._apply_ecc(raw_frame)
        elif self.method == "sparseOptFlow":
            return self._apply_sparseoptflow(raw_frame)
        else:
            return np.eye(2, 3)

    def _apply_ecc(self, raw_frame):
        h, w, c = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if c == 3 else raw_frame
        H = np.eye(2, 3, dtype=np.float32)
        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (w // self.downscale, h // self.downscale))
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.initializedFirstFrame = True
            return H
        try:
            (_, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except Exception as e:
            logger.warning(f"findTransformECC failed; using identity warp. {e}")
        return H

    def _apply_features(self, raw_frame, detections=None):
        h, w, c = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if c == 3 else raw_frame
        H = np.eye(2, 3)
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (w // self.downscale, h // self.downscale))
            w, h = w // self.downscale, h // self.downscale
        mask = np.zeros_like(frame)
        mask[int(0.02 * h):int(0.98 * h), int(0.02 * w):int(0.98 * w)] = 255
        if detections is not None:
            for det in detections:
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                mask[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]] = 0
        keypoints = self.detector.detect(frame, mask)
        keypoints, descriptors = self.extractor.compute(frame, keypoints)
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            self.initializedFirstFrame = True
            return H
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)
        matches, spatialDistances = [], []
        maxSpatialDistance = 0.25 * np.array([w, h])
        if len(knnMatches) == 0:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H
        for m, n in knnMatches:
            if m.distance < 0.9 * n.distance:
                prev = self.prevKeyPoints[m.queryIdx].pt
                curr = keypoints[m.trainIdx].pt
                sd = (prev[0] - curr[0], prev[1] - curr[1])
                if abs(sd[0]) < maxSpatialDistance[0] and abs(sd[1]) < maxSpatialDistance[1]:
                    spatialDistances.append(sd)
                    matches.append(m)
        meanSD = np.mean(spatialDistances, 0)
        stdSD = np.std(spatialDistances, 0)
        inliers = (spatialDistances - meanSD) < 2.5 * stdSD
        prevPoints, currPoints = [], []
        for i in range(len(matches)):
            if inliers[i, 0] and inliers[i, 1]:
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)
        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)
        if prevPoints.shape[0] > 4:
            H, _ = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            logger.warning("not enough matching points")
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)
        return H

    def _apply_sparseoptflow(self, raw_frame):
        h, w, c = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY) if c == 3 else raw_frame
        H = np.eye(2, 3)
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (w // self.downscale, h // self.downscale))
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)
        if not self.initializedFirstFrame or self.prevKeyPoints is None:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.initializedFirstFrame = True
            return H
        matchedKeypoints, status, _ = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)
        prevPoints, currPoints = [], []
        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])
        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)
        if prevPoints.shape[0] > 4 and prevPoints.shape[0] == currPoints.shape[0]:
            H, _ = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            logger.warning("not enough matching points")
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        return H

    def reset_params(self) -> None:
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False
