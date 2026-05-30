"""
filter_engine.py

Core orchestration engine (OPTIMIZED with tracking + downscaling + recognition interval).

Responsibilities:
- Take a single image/frame
- Downscale frame for detection & recognition
- Detect all faces
- Track faces across frames
- Run recognition ONLY for new tracks or periodically
- Blur all non-master faces
- Return processed frame

Standalone usage:
python -m pipeline.filter_engine --master data/master/master.jpg --image data/test_videos/public_vlogger.jpg
"""

import argparse
import cv2

from models.detector import FaceDetector
from models.recognizer import FaceRecognizer
from utils.preprocessing import align_and_crop
from utils.tracker import CentroidTracker


class FilterEngine:
    def __init__(self,
                 threshold=0.45,
                 blur_strength=99,
                 detect_scale=0.45,
                 recognize_interval=10):
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.tracker = CentroidTracker()

        self.threshold = threshold
        self.blur_strength = blur_strength
        self.detect_scale = detect_scale
        self.recognize_interval = recognize_interval
        self.blur_mode = 'gaussian' # or 'black'


        self.master_embedding = None
        self.frame_idx = 0

    # --------------------------------------------------
    # Master setup
    # --------------------------------------------------
    def set_master(self, master_image_bgr):
        faces = self.detector.detect(master_image_bgr)
        if not faces:
            raise RuntimeError('No face detected in master image')

        aligned = align_and_crop(master_image_bgr, faces[0]['kps'])
        if aligned is None:
            raise RuntimeError('Failed to align master face')

        emb = self.recognizer.get_embedding(aligned)
        if emb is None:
            raise RuntimeError('Failed to generate master embedding')

        self.master_embedding = emb
        print('[INFO] Master face registered')

    # --------------------------------------------------
    # Frame processing (TRACKING + CACHING + INTERVAL)
    # --------------------------------------------------
    def process_frame(self, frame_bgr, debug=False):
        if self.master_embedding is None:
            raise RuntimeError('Master embedding not set')

        self.frame_idx += 1
        h, w = frame_bgr.shape[:2]
        scale = self.detect_scale

        # --- Downscale frame for ML ---
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))

        detections = self.detector.detect(small)

        # --- Rescale detections back ---
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            det['bbox'] = (
                int(x1 / scale), int(y1 / scale),
                int(x2 / scale), int(y2 / scale)
            )
            if det['kps'] is not None:
                det['kps'] = (det['kps'] / scale).astype(int)

        tracks = self.tracker.update(detections)
        output = frame_bgr.copy()

        for track in tracks:
            x1, y1, x2, y2 = track.bbox

            # Decide if recognition should run
            need_recognition = (
                track.identity is None or
                (self.frame_idx % self.recognize_interval == 0)
            )

            if need_recognition:
                matched_det = None
                for det in detections:
                    if det['bbox'] == track.bbox:
                        matched_det = det
                        break

                if matched_det and matched_det['kps'] is not None:
                    aligned = align_and_crop(frame_bgr, matched_det['kps'])
                    if aligned is not None:
                        emb = self.recognizer.get_embedding(aligned)
                        if emb is not None:
                            is_master, dist = self.recognizer.is_match(
                                emb, self.master_embedding, self.threshold
                            )
                            track.identity = 'MASTER' if is_master else 'OTHER'
                            track.distance = dist

            # Blur decision
            is_master = (track.identity == 'MASTER')
            if not is_master:
                roi = output[y1:y2, x1:x2]
                if roi.size > 0:
                    if self.blur_mode == 'gaussian':
                        k = max(1, self.blur_strength | 1)
                        blurred = cv2.GaussianBlur(roi, (k, k), 0)
                        output[y1:y2, x1:x2] = blurred

                    elif self.blur_mode == 'black':
                        output[y1:y2, x1:x2] = 0
                        
            if debug:
                color = (0, 255, 0) if is_master else (0, 0, 255)
                label = track.identity if track.identity else 'UNKNOWN'
                if hasattr(track, 'distance'):
                    label += f" {track.distance:.2f}"
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return output


# --------------------------------------------------
# Standalone test runner
# --------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test filter engine (optimized interval)')
    parser.add_argument('--master', '-m', required=True)
    parser.add_argument('--image', '-i', required=True)
    parser.add_argument('--threshold', '-t', type=float, default=0.45)
    args = parser.parse_args()

    master_img = cv2.imread(args.master)
    test_img = cv2.imread(args.image)

    if master_img is None or test_img is None:
        raise RuntimeError('Could not read images')

    engine = FilterEngine(threshold=args.threshold)
    engine.set_master(master_img)

    output = engine.process_frame(test_img, debug=True)

    cv2.imshow('Filter Engine Output (Interval)', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
