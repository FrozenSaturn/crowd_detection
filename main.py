import cv2
from ultralytics import YOLO
import os
import numpy as np
import pandas as pd
import uuid

VIDEO_NAME = "dataset_video.mp4"
INPUT_VIDEO_PATH = os.path.join("input_videos", VIDEO_NAME)
OUTPUT_VIDEO_PATH = os.path.join("output_data", f"tuned_appearance_tracking_{VIDEO_NAME}")
CSV_OUTPUT_PATH = os.path.join("output_data", "tuned_crowd_events_appearance.csv")

MODEL_NAME = 'yolov8m.pt'
YOLO_IMG_SIZE = 416

MIN_PERSONS_FOR_CROWD = 3
CLOSENESS_THRESHOLD_PIXELS = 80
IOU_MATCHING_THRESHOLD = 0.15
MAX_CENTROID_DISTANCE_FOR_IOU_MATCH = 120
PERSON_COUNT_MATCHING_TOLERANCE = 3
MAX_FRAMES_TO_KEEP_LOST_TRACK = 5
MIN_HIST_CORRELATION_THRESHOLD = 0.7
HIST_HUE_BINS = 30
HIST_SAT_BINS = 32
CONSECUTIVE_FRAMES_THRESHOLD = 10

try:
    print(f"Loading YOLO model: {MODEL_NAME} with image size: {YOLO_IMG_SIZE}")
    model = YOLO(MODEL_NAME)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# --- Helper Functions ---
def extract_color_histogram(frame, bbox, hue_bins=HIST_HUE_BINS, sat_bins=HIST_SAT_BINS):
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2: return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return None
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_roi], [0, 1], None, [hue_bins, sat_bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist

def calculate_average_group_histogram(frame, group_members_data, hue_bins=HIST_HUE_BINS, sat_bins=HIST_SAT_BINS):
    if not group_members_data: return None
    member_histograms = [hist for person_data in group_members_data 
                         if (hist := extract_color_histogram(frame, person_data['bbox'], hue_bins, sat_bins)) is not None]
    if not member_histograms: return None
    avg_hist = np.mean(np.array(member_histograms), axis=0).astype(np.float32)
    cv2.normalize(avg_hist, avg_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return avg_hist

def calculate_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def are_persons_close(person1_centroid, person2_centroid, threshold):
    return np.linalg.norm(np.array(person1_centroid) - np.array(person2_centroid)) < threshold

def calculate_group_centroid(group_members_data):
    if not group_members_data: return None
    return (int(np.mean([p['centroid'][0] for p in group_members_data])),
            int(np.mean([p['centroid'][1] for p in group_members_data])))

def calculate_encompassing_bbox(group_members_data):
    if not group_members_data: return None
    return (min(p['bbox'][0] for p in group_members_data), min(p['bbox'][1] for p in group_members_data),
            max(p['bbox'][2] for p in group_members_data), max(p['bbox'][3] for p in group_members_data))

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator != 0 else 0.0

def find_potential_groups(persons_data, closeness_threshold, min_persons):
    if len(persons_data) < min_persons: return []
    num_persons = len(persons_data)
    adj = [[] for _ in range(num_persons)]
    for i in range(num_persons):
        for j in range(i + 1, num_persons):
            if are_persons_close(persons_data[i]['centroid'], persons_data[j]['centroid'], closeness_threshold):
                adj[i].append(j); adj[j].append(i)
    potential_groups = []
    visited = [False] * num_persons
    for i in range(num_persons):
        if not visited[i]:
            q = [i]; visited[i] = True; head = 0
            current_group_indices = []
            while head < len(q):
                u = q[head]; head += 1; current_group_indices.append(u)
                for v_neighbor in adj[u]:
                    if not visited[v_neighbor]: visited[v_neighbor] = True; q.append(v_neighbor)
            if len(current_group_indices) >= min_persons:
                potential_groups.append([persons_data[idx] for idx in current_group_indices])
    return potential_groups

# --- Main Processing Logic ---
def process_video(video_path, output_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Error: Could not open video file {video_path}"); return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    active_tracked_groups = [] 
    confirmed_crowd_log = []
    print("Processing video with tuned appearance-based tracking...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_number += 1
        if frame_number % 50 == 0: print(f"Processing frame {frame_number}...")

        # 1. Detect Persons
        results = model(frame, imgsz=YOLO_IMG_SIZE, classes=[0], verbose=False)
        current_frame_persons_data = [{'bbox': tuple(map(int, box.xyxy[0])), 
                                       'centroid': calculate_centroid(tuple(map(int, box.xyxy[0]))),
                                       'confidence': float(box.conf[0]), 
                                       'frame_number': frame_number,
                                       'id': f"f{frame_number}_p{res_idx}_{box_idx}"}
                                      for res_idx, result in enumerate(results) 
                                      for box_idx, box in enumerate(result.boxes)]

        # --- Draw green outline for ALL detected persons FIRST ---
        for person_data in current_frame_persons_data:
            x1, y1, x2, y2 = person_data['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1) # Thin green outline

        # 2. Identify Potential Groups in Current Frame
        potential_groups_members = find_potential_groups(current_frame_persons_data, 
                                                         CLOSENESS_THRESHOLD_PIXELS, 
                                                         MIN_PERSONS_FOR_CROWD)
        current_frame_group_info = []
        for members in potential_groups_members:
            bbox = calculate_encompassing_bbox(members)
            if bbox:
                current_frame_group_info.append({
                    'members': members, 'group_centroid': calculate_group_centroid(members),
                    'group_bbox': bbox, 'person_count': len(members),
                    'avg_hist': calculate_average_group_histogram(frame, members), 
                    'matched_to_active_group': False})

        # 3. Match Current Frame Groups with Active Tracked Groups
        for ag in active_tracked_groups: ag['updated_this_frame'] = False
        for active_group in active_tracked_groups:
            best_match_cg_data, max_score = None, -1
            ag_bbox, ag_centroid, ag_count, ag_hist = (active_group.get(k) for k in 
                ['group_bbox', 'group_centroid', 'person_count', 'avg_hist'])
            if not all([ag_bbox, ag_centroid, ag_hist is not None]): continue

            for cg_data in current_frame_group_info:
                if cg_data['matched_to_active_group']: continue
                cg_bbox, cg_centroid, cg_count, cg_hist = (cg_data.get(k) for k in 
                    ['group_bbox', 'group_centroid', 'person_count', 'avg_hist'])
                if not all([cg_bbox, cg_centroid, cg_hist is not None]): continue

                iou = calculate_iou(ag_bbox, cg_bbox)
                if iou >= IOU_MATCHING_THRESHOLD:
                    dist = np.linalg.norm(np.array(ag_centroid) - np.array(cg_centroid))
                    hist_corr = cv2.compareHist(ag_hist, cg_hist, cv2.HISTCMP_CORREL)
                    if dist < MAX_CENTROID_DISTANCE_FOR_IOU_MATCH and \
                       abs(ag_count - cg_count) <= PERSON_COUNT_MATCHING_TOLERANCE and \
                       hist_corr >= MIN_HIST_CORRELATION_THRESHOLD and iou > max_score:
                        max_score, best_match_cg_data = iou, cg_data
            
            if best_match_cg_data:
                active_group.update({k: best_match_cg_data[k] for k in 
                                     ['members', 'group_centroid', 'group_bbox', 'person_count', 'avg_hist']})
                active_group.update({'frames_persisted': active_group['frames_persisted'] + 1,
                                     'last_seen_frame': frame_number, 'updated_this_frame': True, 'frames_lost': 0})
                best_match_cg_data['matched_to_active_group'] = True

        # Handle unmatched active groups
        for i in range(len(active_tracked_groups) - 1, -1, -1):
            if not active_tracked_groups[i]['updated_this_frame']:
                active_tracked_groups[i]['frames_lost'] += 1
                if active_tracked_groups[i]['frames_lost'] > MAX_FRAMES_TO_KEEP_LOST_TRACK:
                    active_tracked_groups.pop(i)

        # 4. Add New Unmatched Groups
        for cg_data in current_frame_group_info:
            if not cg_data['matched_to_active_group']:
                active_tracked_groups.append({**cg_data, 'id': str(uuid.uuid4()), 'frames_persisted': 1, 
                                             'last_seen_frame': frame_number, 'updated_this_frame': True,
                                             'first_detected_frame_of_streak': frame_number, 
                                             'is_confirmed_crowd': False, 'frames_lost': 0})

        # 5. Log Confirmed Crowds and Visualize Group Members
        for group in active_tracked_groups:
            if group['frames_lost'] > 0: continue
            viz_color = (0,0,255) if group['frames_persisted'] >= CONSECUTIVE_FRAMES_THRESHOLD else (255,165,0) # Red for crowd, Orange for tracked
            
            if group['frames_persisted'] >= CONSECUTIVE_FRAMES_THRESHOLD:
                if not group.get('is_confirmed_crowd_in_log_for_this_frame', False) or \
                   group.get('logged_person_count') != group['person_count']:
                    confirmed_crowd_log.append({'Frame Number': frame_number, 
                                                'Person Count in Crowd': group['person_count']})
                    group.update({'is_confirmed_crowd_in_log_for_this_frame': True, 
                                  'logged_person_count': group['person_count'], 'is_confirmed_crowd': True})
            elif group['is_confirmed_crowd']: group['is_confirmed_crowd'] = False
            
            if group['frames_persisted'] < CONSECUTIVE_FRAMES_THRESHOLD: # Reset log flag if not a crowd
                group['is_confirmed_crowd_in_log_for_this_frame'] = False
            
            # Draw outlines for members of tracked/confirmed groups (will draw over the initial green lines)
            for p_data in group['members']:
                x1, y1, x2, y2 = p_data['bbox']
                cv2.rectangle(frame, (x1,y1), (x2,y2), viz_color, 2) # Thicker line for group members
                cv2.putText(frame, f"G:{group['id'][:4]} C:{group['person_count']} P:{group['frames_persisted']} L:{group['frames_lost']}", 
                            (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, viz_color, 1)
        
        out.write(frame)
        cv2.imshow('Crowd Tracking - Final Tuned', frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): print("Processing stopped by user."); break

    cap.release(); out.release(); cv2.destroyAllWindows()
    print(f"Finished video processing. Processed video saved to: {output_path}")
    if confirmed_crowd_log:
        pd.DataFrame(confirmed_crowd_log).to_csv(csv_path, index=False)
        print(f"Crowd event log saved to: {csv_path}")
    else:
        print("No crowd events logged.")

if __name__ == "__main__":
    os.makedirs("output_data", exist_ok=True)
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: Input video not found at {INPUT_VIDEO_PATH}")
    else:
        process_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, CSV_OUTPUT_PATH)
