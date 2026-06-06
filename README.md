# FaceBlur

FaceBlur is a lightweight Python project that detects faces in video or camera streams, recognizes a single registered "master" face, and blurs (or blacks out) all other faces. It is designed for performance with several practical optimizations (frame skipping, downscaling for ML, async capture/processing pipeline and a simple tracker + recognition caching).


## Key features

- Detect faces and facial landmarks using InsightFace (RetinaFace)
- Generate ArcFace embeddings and compare against a single master face
- Blur or blackout non-master faces in video or live camera input
- Multiple runners: synchronous, async, GPU-aware async, live camera
- Performance optimizations: frame skipping, detection downscale, recognition interval, lightweight centroid tracker


## Tech stack / dependencies

Primary technologies used in the project:

- Python 3.8+ (recommended 3.8–3.12)
- OpenCV (cv2) — image I/O, display, basic image ops
- numpy — numeric operations
- InsightFace — RetinaFace detection + ArcFace recognition (via FaceAnalysis)
- onnxruntime / onnxruntime-gpu — optional runtime backend for InsightFace models (GPU acceleration)

Recommended pip packages (example; include in your venv):

- opencv-python
- numpy
- insightface
- onnxruntime (or onnxruntime-gpu for NVIDIA GPU)

Note: This repository's `requirements.txt` is currently empty. See the Installation section for an example requirements block.


## Repository layout (brief)

- app/ — high-level entry points (runners): `main.py`, `main_async.py`, `main_gpu.py`, `main_camera.py`
- pipeline/ — `filter_engine.py` orchestrates detection, tracking, recognition and blurring
- models/ — model wrappers: `detector.py` (RetinaFace wrapper) and `recognizer.py` (ArcFace wrapper)
- utils/ — helpers: `preprocessing.py` (alignment & crop), `tracker.py` (centroid tracker), `video.py`, `drawing.py`
- data/ — sample master images and test videos
- output_*.mp4 — example outputs produced by past runs


## How it works (high-level)

1. A master image containing the authorized person's face is registered. The pipeline extracts and stores an ArcFace embedding for that face.
2. For each frame in a video or camera stream the pipeline:
   - Optionally downscales the frame for detection to reduce CPU cost
   - Runs RetinaFace to get bounding boxes + 5-point landmarks
   - Rescales detections back to original frame size
   - Tracks faces across frames using a simple centroid tracker (maintains persistent track IDs)
   - Runs recognition (ArcFace embedding + cosine distance) only when a track is new or periodically (recognize interval) to save compute
   - Marks tracks as MASTER or OTHER based on distance to the registered master embedding
   - Blurs or blacks out regions belonging to non-master faces and returns the processed frame
3. Runners handle I/O and optional async threading, writing outputs and showing the display window.


## Installation (Windows example)

1. Create a virtual environment and activate it (PowerShell):

    python -m venv .venv; .\.venv\Scripts\Activate.ps1

2. Upgrade pip and install dependencies:

    python -m pip install --upgrade pip

Create a `requirements.txt` (example contents):

    numpy
    opencv-python
    insightface
    onnxruntime

Then install:

    pip install -r requirements.txt

If you have an NVIDIA GPU and want to use GPU acceleration, install `onnxruntime-gpu` instead of `onnxruntime` (and ensure compatible CUDA/cuDNN are installed):

    pip uninstall onnxruntime; pip install onnxruntime-gpu

Notes:
- InsightFace (FaceAnalysis) will download model files (e.g. `buffalo_l`) on first run and cache them. Allow the environment network access on first execution.
- If you prefer CPU-only operation, keep `onnxruntime` (CPU) and InsightFace will use the CPU provider.


## Quick start / Usage

All runners expect a "master" image containing a single face to be used as the identity that should NOT be blurred.

Basic synchronous video runner (simple, single-threaded):

    python -m app.main --master data/master/master.png --video data/test_videos/TrimmedVlog.mp4 --output output_blurred.mp4 --skip 2

Async video runner (decouples capture and ML processing, better throughput):

    python -m app.main_async --master data/master/master.png --video data/test_videos/TrimmedVlog.mp4 --output output_blurred_async.mp4

GPU-aware async runner (attempts to prefer GPU providers):

    python -m app.main_gpu --master data/master/master.png --video data/test_videos/TrimmedVlog.mp4 --output output_blurred_gpu.mp4 --skip 2

Live camera (webcam) runner:

    python -m app.main_camera --master data/master/master.png --camera 0 --skip 2

Common CLI options (per runner):

- --master / -m : path to master image (required)
- --video / -v  : path to input video (for video runners)
- --output / -o : optional output video path (mp4)
- --threshold / -t : recognition distance threshold (default ~0.45)
- --skip : process every Nth frame (default 1; use >1 to speed up)
- --debug : draw bounding boxes + labels on output windows
- --queue-size : (async runners) queue capacity between threads
- --display-scale : (main) scale display window only

Examples of tuning for performance:
- Increase --skip to 2–4 to process fewer frames (reduces CPU usage)
- Increase detect downscale (in code: `detect_scale` in `FilterEngine`) to lower detection cost at the expense of accuracy
- Increase `recognize_interval` (in `FilterEngine`) to run recognition less often per track


## Running individual components / tests

- Test the detector directly:

    python -m models.detector --image data/test_videos/public_vlogger.jpg

- Test the recognizer (generates a master embedding and tries to match faces in a test image):

    python -m models.recognizer --master data/master/master.jpg --image data/test_videos/public_vlogger.jpg

- Test the filter engine with a static image:

    python -m pipeline.filter_engine --master data/master/master.jpg --image data/test_videos/public_vlogger.jpg


## Notes, caveats and tips

- Master image: Use a clear, frontal photo with a single face. If multiple faces are present the first detected face will be used as master.
- Thresholds: The default threshold (~0.45) is sensible for ArcFace embeddings but may need adjustment depending on lighting, camera, and model variants.
- Models: InsightFace's `buffalo_l` bundle includes both RetinaFace and ArcFace models and will be downloaded to the InsightFace cache on first run. The `models/` folder contains an additional `yolov8n-face.pt` file — it is not used by the current InsightFace-based detector but may be part of earlier experiments.
- GPU: To leverage GPU acceleration, install `onnxruntime-gpu` and ensure the correct CUDA / cuDNN runtime is installed. `app/main_gpu.py` logs available ONNX providers and will fall back to CPU when necessary.
- Privacy / ethics: This tool modifies video frames and could be used to obscure identities. Ensure you have proper consent and legal basis for processing video and faces.


## Contributing

- Follow standard Python packaging & linting practices.
- Add missing dependencies to `requirements.txt` if you introduce new libs.
- Consider adding automated tests for `utils/tracker.py` and `utils/preprocessing.py` for regressions.


## License

This repository does not include a license file. Add an appropriate open-source license if you intend to publish or share the project.



--
Generated README: concise project overview, installation and multiple run examples. If you want, I can also add a populated `requirements.txt` and example PowerShell scripts to automate setup.