"""
main_camera.py

Real-time camera runner for FaceBlur (CPU / GPU compatible).

Features:
- Live webcam input
- Async pipeline (capture + processing + output)
- Supports --skip for FPS control
- Works with existing FilterEngine (no code duplication)

Run:
python -m app.main_camera \
  --master data/master/master3.png \
  --camera 0 \
  --skip 2
"""

import argparse
import cv2
import time
import threading
import queue

from pipeline.filter_engine import FilterEngine


def main():
    parser = argparse.ArgumentParser(description='FaceBlur live camera runner')
    parser.add_argument('--master', '-m', required=True, help='Path to master image')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
    parser.add_argument('--threshold', '-t', type=float, default=0.45)
    parser.add_argument('--skip', type=int, default=1, help='Process every Nth frame')
    parser.add_argument('--queue-size', type=int, default=5)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # ----------------------
    # Load master image
    # ----------------------
    master_img = cv2.imread(args.master)
    if master_img is None:
        raise RuntimeError('Could not read master image')

    # ----------------------
    # Open camera
    # ----------------------
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera index {args.camera}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f'[INFO] Camera opened: {width}x{height} @ {fps:.2f} FPS')

    # ----------------------
    # Initialize filter engine
    # ----------------------
    engine = FilterEngine(threshold=args.threshold)
    engine.set_master(master_img)

    # ----------------------
    # Queues & threading
    # ----------------------
    frame_q = queue.Queue(maxsize=args.queue_size)
    result_q = queue.Queue(maxsize=args.queue_size)
    stop_flag = threading.Event()

    # ----------------------
    # Capture thread
    # ----------------------
    def capture_loop():
        frame_idx = 0
        while not stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                stop_flag.set()
                break

            frame_idx += 1
            if frame_idx % args.skip != 0:
                continue

            try:
                frame_q.put(frame, timeout=0.01)
            except queue.Full:
                pass

    # ----------------------
    # Processing thread
    # ----------------------
    def process_loop():
        while not stop_flag.is_set() or not frame_q.empty():
            try:
                frame = frame_q.get(timeout=0.05)
            except queue.Empty:
                continue

            output = engine.process_frame(frame, debug=args.debug)
            try:
                result_q.put(output, timeout=0.05)
            except queue.Full:
                pass

    # Start threads
    t_cap = threading.Thread(target=capture_loop, daemon=True)
    t_proc = threading.Thread(target=process_loop, daemon=True)
    t_cap.start()
    t_proc.start()

    # ----------------------
    # Display loop
    # ----------------------
    frames_out = 0
    t0 = time.time()

    while not stop_flag.is_set() or not result_q.empty():
        try:
            output = result_q.get(timeout=0.05)
        except queue.Empty:
            continue

        cv2.imshow('FaceBlur - Live Camera', output)
        frames_out += 1

        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            stop_flag.set()
            break

    # ----------------------
    # Cleanup
    # ----------------------
    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f'[INFO] Frames displayed: {frames_out}')
    print(f'[INFO] Effective FPS: {frames_out / elapsed:.2f}')


if __name__ == '__main__':
    main()
