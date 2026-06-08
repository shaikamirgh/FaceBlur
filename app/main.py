"""
main.py

High-performance video runner for FaceBlur (OPTIMIZED with multi-threading).

Optimizations:
- Frame skipping (process every Nth frame)
- FPS-aware display
- Multi-threaded I/O (background reader/writer)
- Aggressive downscaling (30% detection scale)
- Detection caching (every 3rd frame)
- Skip tiny faces (< 20px)

Run:
python -m app.main \
  --master data/master/master.jpg \
  --video data/test_videos/input.mp4 \
  --output output_blurred.mp4 \
  --skip 2
"""

import argparse
import cv2
import os
import time
import threading
import queue
from collections import deque

from pipeline.filter_engine import FilterEngine


class FrameReader(threading.Thread):
    """Background thread that reads frames from video."""
    def __init__(self, video_path, output_queue, max_queue_size=30):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.output_queue = output_queue
        self.max_queue_size = max_queue_size
        self.stop_flag = False
        self.frame_count = 0
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.output_queue.put(None)
            return
        
        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                self.output_queue.put(None)
                break
            
            # Don't block if queue is full
            try:
                self.output_queue.put(frame, timeout=0.1)
                self.frame_count += 1
            except queue.Full:
                pass
        
        cap.release()


class FrameWriter(threading.Thread):
    """Background thread that writes frames to output video."""
    def __init__(self, output_path, fps, width, height, input_queue):
        super().__init__(daemon=True)
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.input_queue = input_queue
        self.stop_flag = False
        self.frame_count = 0
        
    def run(self):
        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        while not self.stop_flag:
            try:
                frame = self.input_queue.get(timeout=1.0)
                if frame is None:
                    break
                writer.write(frame)
                self.frame_count += 1
            except queue.Empty:
                pass
        
        writer.release()


def main():
    parser = argparse.ArgumentParser(description='FaceBlur video processor (optimized)')
    parser.add_argument('--master', '-m', required=True, help='Path to master image')
    parser.add_argument('--video', '-v', required=True, help='Path to input video')
    parser.add_argument('--output', '-o', default=None, help='Optional output video path')
    parser.add_argument('--threshold', '-t', type=float, default=0.45)
    parser.add_argument('--skip', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--display-scale', type=float, default=1.0, help='Resize display window only')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # ---- Load master image ----
    master_img = cv2.imread(args.master)
    if master_img is None:
        raise RuntimeError(f'Could not read master image: {args.master}')

    # ---- Load video ----
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {args.video}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 25
    cap.release()

    print(f'[INFO] Video loaded: {width}x{height} @ {fps:.2f} FPS')

    # ---- Initialize filter engine ----
    engine = FilterEngine(threshold=args.threshold)
    engine.set_master(master_img)

    frame_idx = 0
    processed_frames = 0
    last_output = None
    t0 = time.time()

    # ---- Setup I/O threading ----
    frame_queue = queue.Queue(maxsize=130)
    output_queue = queue.Queue(maxsize=130)
    
    reader = FrameReader(args.video, frame_queue)
    writer = None
    if args.output:
        writer = FrameWriter(args.output, fps, width, height, output_queue)
        writer.start()
        print(f'[INFO] Writing output to: {args.output}')
    
    reader.start()

    # ---- Main loop ----
    try:
        while True:
            try:
                frame = frame_queue.get(timeout=2.0)
            except queue.Empty:
                break
            
            if frame is None:
                break

            frame_idx += 1

            # --- Skip frames (HUGE FPS WIN) ---
            if frame_idx % args.skip == 0:
                last_output = engine.process_frame(frame, debug=args.debug)
                processed_frames += 1
            else:
                # Reuse last processed frame
                if last_output is None:
                    last_output = frame

            display = last_output

            # --- Display scaling (UI only) ---
            if args.display_scale != 1.0:
                display = cv2.resize(
                    display,
                    (int(width * args.display_scale), int(height * args.display_scale))
                )

            try:
                elapsed = time.time() - t0
                fps_display = processed_frames / max(1e-6, elapsed)
                cv2.putText(display, f'FPS: {fps_display:.2f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            except Exception:
                pass

            cv2.imshow('FaceBlur - Video', display)

            if writer is not None and last_output is not None:
                try:
                    output_queue.put_nowait(last_output)
                except queue.Full:
                    pass

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        reader.stop_flag = True
        reader.join(timeout=5)

        if writer is not None:
            try:
                output_queue.put(None, timeout=1.0)
            except queue.Full:
                writer.stop_flag = True
            writer.join(timeout=10)

        cv2.destroyAllWindows()

        elapsed = time.time() - t0
        print(f'[INFO] Frames read: {frame_idx}')
        print(f'[INFO] Frames processed: {processed_frames}')
        print(f'[INFO] Effective processing FPS: {processed_frames / elapsed:.2f}')


if __name__ == '__main__':
    main()
