"""
tracker.py

Simple centroid-based face tracker with identity caching.

Responsibilities:
- Assign persistent IDs to detected face bounding boxes
- Track faces across frames using centroid distance
- Cache identity decision (MASTER / OTHER) per track

Design goals:
- Very lightweight (CPU-friendly)
- No external dependencies
- Easy to reason about and debug

This tracker is intentionally simple and sufficient for MVP.

Standalone test:
python -m utils.tracker
"""

import math


class Track:
    def __init__(self, track_id, bbox, identity=None):
        self.id = track_id
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.identity = identity  # 'MASTER' | 'OTHER' | None
        self.missed = 0

    def centroid(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class CentroidTracker:
    def __init__(self, max_distance=50, max_missed=5):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 0
        self.tracks = {}

    def _distance(self, c1, c2):
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections: list of dicts with key 'bbox'

        Returns:
            list of Track objects (active tracks)
        """
        # Case 1: no existing tracks
        if not self.tracks:
            for det in detections:
                self._create_track(det['bbox'])
            return list(self.tracks.values())

        used_tracks = set()

        # Match detections to existing tracks
        for det in detections:
            det_centroid = self._centroid(det['bbox'])

            best_id = None
            best_dist = None

            for tid, track in self.tracks.items():
                if tid in used_tracks:
                    continue

                dist = self._distance(det_centroid, track.centroid())
                if dist < self.max_distance and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                track = self.tracks[best_id]
                track.bbox = det['bbox']
                track.missed = 0
                used_tracks.add(best_id)
            else:
                self._create_track(det['bbox'])

        # Handle missed tracks
        to_delete = []
        for tid, track in self.tracks.items():
            if tid not in used_tracks:
                track.missed += 1
                if track.missed > self.max_missed:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        return list(self.tracks.values())

    def _create_track(self, bbox):
        self.tracks[self.next_id] = Track(self.next_id, bbox)
        self.next_id += 1

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


# --------------------------------------------------
# Standalone test runner (synthetic test)
# --------------------------------------------------
if __name__ == '__main__':
    print('[TEST] Running CentroidTracker standalone test')

    tracker = CentroidTracker(max_distance=60, max_missed=2)

    # Simulated detections across frames
    frames = [
        [{'bbox': (100, 100, 160, 160)}, {'bbox': (300, 100, 360, 160)}],
        [{'bbox': (105, 105, 165, 165)}, {'bbox': (305, 105, 365, 165)}],
        [{'bbox': (110, 110, 170, 170)}],  # second face disappears
        [{'bbox': (115, 115, 175, 175)}],
        []
    ]

    for frame_idx, detections in enumerate(frames):
        tracks = tracker.update(detections)
        print(f'Frame {frame_idx}:')
        for t in tracks:
            print(f'  Track {t.id}: bbox={t.bbox}, missed={t.missed}')
