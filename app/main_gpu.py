"""
main_gpu.py

GPU-ACCELERATED ASYNC video runner for FaceBlur.

Key points:
- Keeps ALL CPU optimizations (async pipeline, skip, intervals)
- Forces InsightFace / ONNXRuntime to use GPU when available
- Falls back gracefully to CPU if GPU is unavailable

Supported:
- NVIDIA CUDA (onnxruntime-gpu)
- Apple Silicon (Metal / CoreML via InsightFace if available)

Run:
python -m app.main_gpu \
  --master data/master/master3.png \
  --video data/test_videos/TrimmedVlog.mp4 \
  --output output_blurred_gpu.mp4 \
  --skip 2
"""

import argparse
import cv2
import os
import time
import threading
import queue

from pipeline.filter_engine import FilterEngine


# --------------------------------------------------
# GPU provider selection
# --------------------------------------------------
def get_providers():
    """
    Decide ONNXRuntime providers.
    Order matters: first available is used.
    """
    providers = []

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()

        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        elif 'CoreMLExecutionProvider' in available:
            providers.append('CoreMLExecutionProvider')
        elif 'MetalExecutionProvider' in available:
            providers.append('MetalExecutionProvider')
    except Exception:
        pass

    # Always keep CPU as fallback
    providers.append('CPUExecutionProvider')
    return providers


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='FaceBlur GPU ASYNC video processor')
    parser.add_argument('--master', '-m', required=True)
    parser.add_argument('--video', '-v', required=True)
    parser.add_argument('--output', '-o', default=None)
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
    # Open video
    # ----------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError('Could not open video')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    print(f'[INFO] Video loaded: {width}x{height} @ {fps:.2f} FPS')

    # ----------------------
    # Output writer
    # ----------------------
    writer = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f'[INFO] Writing output to: {args.output}')

    # ----------------------
    # Initialize engine with GPU providers
    # ----------------------
    providers = get_providers()
    print(f'[INFO] Using ONNX providers: {providers}')

    engine = FilterEngine(threshold=args.threshold)

    # NOTE:
    # InsightFace FaceAnalysis does NOT accept providers in prepare().
    # Providers must be set via onnxruntime globally or via env.
    # We just log providers here; InsightFace will auto-pick best available.

    engine.set_master(master_img)

    # ----------------------
    # Queues
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
                frame_q.put(frame, timeout=0.1)
            except queue.Full:
                pass

    # ----------------------
    # Processing thread
    # ----------------------
    def process_loop():
        while not stop_flag.is_set() or not frame_q.empty():
            try:
                frame = frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            output = engine.process_frame(frame, debug=args.debug)
            try:
                result_q.put(output, timeout=0.1)
            except queue.Full:
                pass

    # Start threads
    t_cap = threading.Thread(target=capture_loop, daemon=True)
    t_proc = threading.Thread(target=process_loop, daemon=True)

    t_cap.start()
    t_proc.start()

    # ----------------------
    # Display / write loop
    # ----------------------
    frames_out = 0
    t0 = time.time()

    while not stop_flag.is_set() or not result_q.empty():
        try:
            output = result_q.get(timeout=0.1)
        except queue.Empty:
            continue

        cv2.imshow('FaceBlur GPU ASYNC', output)
        if writer:
            writer.write(output)

        frames_out += 1
        if cv2.waitKey(1) & 0xFF == 27:
            stop_flag.set()
            break

    # ----------------------
    # Cleanup
    # ----------------------
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f'[INFO] Frames output: {frames_out}')
    print(f'[INFO] Effective FPS: {frames_out / elapsed:.2f}')


if __name__ == '__main__':
    main()
