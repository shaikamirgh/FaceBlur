"""
main.py

High-performance video runner for FaceBlur (OPTIMIZED).

Optimizations added:
- Frame skipping (process every Nth frame)
- FPS-aware display
- Optional resize for display only
- Clean separation of I/O vs ML

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

from pipeline.filter_engine import FilterEngine


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

    print(f'[INFO] Video loaded: {width}x{height} @ {fps:.2f} FPS')

    # ---- Output writer ----
    writer = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f'[INFO] Writing output to: {args.output}')

    # ---- Initialize filter engine ----
    engine = FilterEngine(threshold=args.threshold)
    engine.set_master(master_img)

    frame_idx = 0
    processed_frames = 0
    last_output = None
    t0 = time.time()

    # ---- Main loop ----
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # --- Skip frames (HUGE FPS WIN) ---
        if frame_idx % args.skip == 0:
            last_output = engine.process_frame(frame, debug=args.debug)
            processed_frames += 1
        else:
            # Reuse last processed frame
            if last_output is not None:
                last_output = last_output
            else:
                last_output = frame

        display = last_output

        # --- Display scaling (UI only) ---
        if args.display_scale != 1.0:
            display = cv2.resize(
                display,
                (int(width * args.display_scale), int(height * args.display_scale))
            )

        try:
            elapsed = time.time() -t0
            fps_display = processed_frames / max(1e-6, elapsed)
            cv2.putText(display, f'FPS: {fps_display:.2f}', (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        except Exception:
            pass

        cv2.imshow('FaceBlur - Video', display)

        if writer is not None:
            writer.write(last_output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # ---- Cleanup ----
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f'[INFO] Frames read: {frame_idx}')
    print(f'[INFO] Frames processed: {processed_frames}')
    print(f'[INFO] Effective processing FPS: {processed_frames / elapsed:.2f}')


if __name__ == '__main__':
    main()
