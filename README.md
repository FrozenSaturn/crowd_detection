# Crowd Detection System

## 1. Project Description

This Python script detects persons in a video, identifies potential crowds (3+ people close together), tracks these groups using spatial and appearance features, and logs confirmed crowd events to a CSV file. It also generates a processed video with visualizations.

## 2. Setup

1.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    ```
2.  **Activate Environment:**
    * Linux/macOS: `source venv/bin/activate`
    * Windows: `venv\Scripts\activate`
3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
    (YOLO model weights are downloaded automatically on the first run.)

## 3. Requirements

All Python package dependencies are listed in `requirements.txt`. Key libraries include:
* OpenCV (`opencv-python`)
* Ultralytics YOLO
* NumPy
* Pandas

## 4. Implementation

### How to Run:
1.  Place the input video (e.g., `dataset_video.mp4`) in the `input_videos/` directory.
2.  Execute `main.py` from the project root:
    ```bash
    python main.py
    ```

### Key Configurable Parameters (in `main.py`):
* `VIDEO_NAME`: Name of the input video file.
* `MODEL_NAME`: YOLO model (default: `'yolov8m.pt'`).
* `YOLO_IMG_SIZE`: YOLO inference image size (default: `416`).
* `MIN_PERSONS_FOR_CROWD`: Min persons for a group (default: `3`).
* `CLOSENESS_THRESHOLD_PIXELS`: For initial person grouping (default: `80`).
* `CONSECUTIVE_FRAMES_THRESHOLD`: For crowd confirmation (default: `10`).
* **Tracking Cues:**
    * `IOU_MATCHING_THRESHOLD`: Min IoU for group bbox matching (default: `0.15`).
    * `MAX_CENTROID_DISTANCE_FOR_IOU_MATCH`: Max centroid distance for valid IoU match (default: `120`).
    * `PERSON_COUNT_MATCHING_TOLERANCE`: Allowed person count difference (default: `3`).
    * `MAX_FRAMES_TO_KEEP_LOST_TRACK`: Grace period for lost tracks (default: `5`).
    * `MIN_HIST_CORRELATION_THRESHOLD`: Min histogram correlation for appearance match (default: `0.7`).
    * `HIST_HUE_BINS`, `HIST_SAT_BINS`: Bins for H-S color histograms (defaults: `30`, `32`).

## 5. Special Mentions

* Utilizes the YOLOv8m model for person detection.
* Employs a multi-cue tracking approach including IoU of group bounding boxes, centroid proximity, person count consistency, appearance (color histograms), and a grace period for improved group ID consistency.

## 6. Output

* **Processed Video:** Saved in `output_data/` (e.g., `tuned_appearance_tracking_dataset_video.mp4`), showing detected persons, tracked groups (orange), and confirmed crowds (red).
* **CSV File:** Saved in `output_data/` (e.g., `tuned_crowd_events_appearance.csv`), logging `Frame Number` and `Person Count in Crowd` for confirmed crowd events.
