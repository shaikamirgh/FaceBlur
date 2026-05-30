"""
preprocessing.py

Face preprocessing utilities.

Responsibilities:
- Take detector output (bbox + keypoints)
- Produce a clean, aligned face crop suitable for ArcFace
- No detection, no recognition, no decisions

This module is intentionally simple and testable on its own.

Standalone usage:
python preprocessing.py --image path/to/image.jpg
"""

import argparse
import cv2
import numpy as np
from models.detector import FaceDetector

# ArcFace standard input size
ARCFACE_SIZE = (112, 112)


def align_and_crop(frame, kps, output_size=ARCFACE_SIZE):
    """
    Align and crop face using 5-point landmarks.

    Args:
        frame (np.ndarray): BGR image
        kps (np.ndarray): shape (5,2) facial landmarks
        output_size (tuple): target (width, height)

    Returns:
        aligned_face (np.ndarray): aligned BGR face image
    """
    # Standard ArcFace reference points (from InsightFace)
    ref_pts = np.array([
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041],  # right mouth
    ], dtype=np.float32)

    ref_pts[:, 0] *= (output_size[0] / 112)
    ref_pts[:, 1] *= (output_size[1] / 112)

    src_pts = kps.astype(np.float32)

    # Estimate similarity transform
    M, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)
    if M is None:
        return None

    aligned = cv2.warpAffine(
        frame,
        M,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderValue=0.0
    )

    return aligned


# --------------------------------------------------
# Standalone test runner
# --------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test face alignment and cropping')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    detector = FaceDetector()
    detections = detector.detect(img)

    print(f"Detected {len(detections)} face(s)")

    for idx, d in enumerate(detections):
        kps = d['kps']
        if kps is None:
            continue

        aligned = align_and_crop(img, kps)
        if aligned is None:
            continue

        cv2.imshow(f"aligned_face_{idx}", aligned)
        cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
