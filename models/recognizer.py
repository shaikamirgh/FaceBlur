"""
recognizer.py

ArcFace-based face recognition module.

Responsibilities:
- Load ArcFace model (via InsightFace)
- Generate embeddings from aligned face crops (112x112)
- Compare embeddings with a master embedding

Design goals:
- Simple, explicit API
- Standalone runnable for testing
- No detection, no blurring, no video logic

Standalone usage:
python -m models.recognizer --master data/test_videos/public_vlogger.jpg --image data/test_videos/public_vlogger.jpg
"""

import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis


# --------------------------------------------------
# Recognizer class
# --------------------------------------------------
class FaceRecognizer:
    def __init__(self, providers=None):
        if providers is None:
            providers = ['CPUExecutionProvider']

        # IMPORTANT:
        # We use FaceAnalysis ONLY to load the ArcFace recognition model.
        # We do NOT call app.get() on aligned crops.
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0)

        self.rec_model = self.app.models.get('recognition')
        if self.rec_model is None:
            raise RuntimeError('ArcFace recognition model not found')

    def get_embedding(self, aligned_face_bgr):
        """
        Generate ArcFace embedding from an aligned face crop.

        Args:
            aligned_face_bgr (np.ndarray): 112x112 BGR face image

        Returns:
            np.ndarray | None: normalized embedding vector
        """
        if aligned_face_bgr is None:
            return None

        if aligned_face_bgr.shape[:2] != (112, 112):
            return None

        # ArcFace expects RGB, CHW, float32
        # Convert BGR -> RGB
        face_rgb = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2RGB)
        # HWC -> CHW
        face_rgb = face_rgb.transpose(2, 0, 1)
        # Add batch dimension: (1, 3, 112, 112)
        face_rgb = np.expand_dims(face_rgb, axis=0).astype(np.float32)

        # Run ArcFace forward pass
        emb = self.rec_model.forward(face_rgb)

        # emb shape: (1, 512) -> flatten to (512,)
        emb = emb.squeeze()
        # L2 normalize (safety)
        emb = emb / np.linalg.norm(emb)
        return emb

    @staticmethod
    def cosine_distance(emb1, emb2):
        return 1.0 - float(np.dot(emb1, emb2))

    def is_match(self, emb, master_emb, threshold=0.45):
        dist = self.cosine_distance(emb, master_emb)
        return dist < threshold, dist


# --------------------------------------------------
# Standalone test runner
# --------------------------------------------------
if __name__ == '__main__':
    from utils.preprocessing import align_and_crop
    from models.detector import FaceDetector

    parser = argparse.ArgumentParser(description='Test ArcFace recognition')
    parser.add_argument('--master', '-m', required=True)
    parser.add_argument('--image', '-i', required=True)
    parser.add_argument('--threshold', '-t', type=float, default=0.45)
    args = parser.parse_args()

    master_img = cv2.imread(args.master)
    test_img = cv2.imread(args.image)
    if master_img is None or test_img is None:
        raise RuntimeError('Could not read images')

    detector = FaceDetector()
    recognizer = FaceRecognizer()

    # ---- Master embedding ----
    master_faces = detector.detect(master_img)
    if not master_faces:
        raise RuntimeError('No face detected in master image')

    master_aligned = align_and_crop(master_img, master_faces[0]['kps'])
    master_emb = recognizer.get_embedding(master_aligned)
    if master_emb is None:
        raise RuntimeError('Failed to generate master embedding')

    print('Master embedding generated successfully')

    # ---- Test image ----
    test_faces = detector.detect(test_img)
    print(f'Detected {len(test_faces)} face(s)')

    matched_idx = None
    matched_bbox = None

    for idx, f in enumerate(test_faces):
        aligned = align_and_crop(test_img, f['kps'])
        if aligned is None:
            print(f'[DEBUG] Face {idx}: alignment failed')
            continue

        emb = recognizer.get_embedding(aligned)
        if emb is None:
            print(f'[DEBUG] Face {idx}: embedding failed')
            continue

        match, dist = recognizer.is_match(emb, master_emb, args.threshold)
        print(f'[DEBUG] Face {idx}: match={match}, distance={dist:.3f}')

        if match and matched_idx is None:
            matched_idx = idx
            matched_bbox = f['bbox']

    # ---- Visualize result ----
    output = test_img.copy()

    if matched_bbox is not None:
        x1, y1, x2, y2 = matched_bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(output, 'MASTER', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(f'[INFO] Master face detected at index {matched_idx}')
    else:
        print('[INFO] No matching master face found')

    cv2.imshow('Recognition Result', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
