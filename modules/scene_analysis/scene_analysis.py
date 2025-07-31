# scene_analyzer.py
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import cv2
import os

def detect_scenes(video_path, threshold=3.0, min_scene_len=5):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))

    video_manager.set_downscale_factor()
    video_manager.start()

    print("[INFO] Detecting scenes...")
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    print(f"[INFO] Total scenes detected: {len(scene_list)}")

    scenes = []
    for i, (start, end) in enumerate(scene_list):
        start_time = start.get_seconds()
        end_time = end.get_seconds()
        scenes.append({'scene': i+1, 'start': start_time, 'end': end_time, 'duration': round(end_time - start_time, 2)})

    video_manager.release()
    return scenes

def extract_representative_frame(video_path, timestamp, output_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    success, frame = cap.read()
    if success:
        cv2.imwrite(output_path, frame)
    else:
        print(f"[WARNING] Could not extract frame at {timestamp:.2f}s")
    cap.release()

if __name__ == "__main__":
    video_path = "D:/bhashasetu_subhashit/input.mp4"
    output_dir = "scene_frames"
    os.makedirs(output_dir, exist_ok=True)

    # Debug info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    print(f"[DEBUG] FPS: {fps}, Frame count: {frame_count}, Duration: {duration:.2f}s")
    cap.release()

    scenes = detect_scenes(video_path, threshold=3.0, min_scene_len=5)

    print("\n[INFO] Extracting frames...")
    for scene in scenes:
        scene_num = scene['scene']
        midpoint = (scene['start'] + scene['end']) / 2
        output_path = os.path.join(output_dir, f"scene_{scene_num}.jpg")
        extract_representative_frame(video_path, midpoint, output_path)
        print(f"[✓] Saved frame: scene_{scene_num}.jpg")
