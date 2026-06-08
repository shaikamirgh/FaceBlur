"""
detector.py

RetinaFace-based face detector module.

Responsibilities:
- Load a pretrained RetinaFace detector (via InsightFace)
- Detect faces in a single image/frame
- Return bounding boxes + 5 facial keypoints

Design goals:
- Simple API
- Can be run standalone for testing/debugging
- No recognition logic here (ONLY detection)

Standalone usage:
python detector.py --image path/to/image.jpg
"""

import argparse
import cv2
from insightface.app import FaceAnalysis


class FaceDetector:
    """
    Thin wrapper around InsightFace RetinaFace.
    """
    def __init__(self, det_size=(640, 640), providers=None):
        if providers is None:
            # GPU-first strategy: try CUDA, fall back to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        # buffalo_l includes RetinaFace + ArcFace models
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, frame_bgr):
        """
        Detect faces in a BGR image.

        Args:
            frame_bgr (np.ndarray): OpenCV BGR image

        Returns:
            list of dicts, each containing:
                - bbox: (x1, y1, x2, y2)
                - kps:  np.ndarray of shape (5, 2)
                - score: detection confidence
        """
        faces = self.app.get(frame_bgr)

        results = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            kps = f.kps.astype(int) if f.kps is not None else None
            score = float(f.det_score)

            results.append({
                'bbox': (x1, y1, x2, y2),
                'kps': kps,
                'score': score
            })

        return results


# --------------------------------------------------
# Standalone test runner
# --------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test RetinaFace detector')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    detector = FaceDetector()
    detections = detector.detect(img)

    print(f"Detected {len(detections)} face(s)")

    # Draw results
    for idx, d in enumerate(detections):
        x1, y1, x2, y2 = d['bbox']
        kps = d['kps']
        score = d['score']

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{score:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if kps is not None:
            for (x, y) in kps:
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow('Detector Test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
