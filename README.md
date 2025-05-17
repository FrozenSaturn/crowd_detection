# Crowd Detection System (Assignment Task 3)

This script detects persons in a video, identifies potential crowds (3+ people close together), tracks them, and logs confirmed crowd events (persisting for 10+ frames) to a CSV file. It also generates a processed video with visualizations.

## Setup

1.  **Create & Activate Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (YOLO model weights are downloaded automatically on first run.)

## Running the Script

1.  Place the input video (e.g., `dataset_video.mp4`) in the `input_videos/` directory.
2.  Execute `main.py` from the project root:
    ```bash
    python main.py
    ```
3.  Outputs appear in `output_data/`:
    * Processed video (e.g., `processed_iou_tracking_dataset_video.mp4`).
    * CSV log of crowd events (e.g., `crowd_events_iou.csv`).

## Key Configuration Parameters (in `main.py`)

* `VIDEO_NAME`: Input video file name.
* `MODEL_NAME`: YOLO model (default: `'yolov8m.pt'`).
* `YOLO_IMG_SIZE`: YOLO inference image size (default: `416`).
* `MIN_PERSONS_FOR_CROWD`: Min persons for a group (default: `3`).
* `CLOSENESS_THRESHOLD_PIXELS`: Max distance for initial person grouping (default: `75`).
* `CONSECUTIVE_FRAMES_THRESHOLD`: Frames a group must persist to be a crowd (default: `10`).
* `IOU_MATCHING_THRESHOLD`: Min IoU for matching groups across frames (default: `0.2`).
* `PERSON_COUNT_MATCHING_TOLERANCE`: Allowed person count difference for group matching (default: `3`).

## Output

* **Processed Video:** Visualizes detected persons, tracked groups (orange), and confirmed crowds (red).
* **CSV File (`crowd_events_iou.csv`):** Logs `Frame Number` and `Person Count in Crowd` for confirmed crowd events.