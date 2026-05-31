import os
import cv2

# 원하는 clip 파일과 추출할 초 지정
# time_sec는 "해당 segment clip 안에서의 시간" 기준
TARGETS = [
    {
        "video_id": "F88btEbzdw0",
        "clip_name": "F88btEbzdw0_op5_action2_0_4.21_13.86.mp4",
        "time_sec": 8.0,
        "output_name": "injection_01.jpg",
    },
    {
        "video_id": "Peyw-eKZJOI",
        "clip_name": "Peyw-eKZJOI_op26_action59_1_134.86_147.56.mp4",
        "time_sec": 5.0,
        "output_name": "bed_shampoo_01.jpg",
    },
]

SEGMENT_DIR = "./dataset/segments"
OUTPUT_DIR = "./dataset/selected_images"


def extract_frame(video_path, output_path, time_sec):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)

    success, frame = cap.read()
    cap.release()

    if not success:
        print(f"[ERROR] Cannot read frame at {time_sec}s: {video_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"[OK] Saved: {output_path}")


for item in TARGETS:
    video_path = os.path.join(
        SEGMENT_DIR,
        item["video_id"],
        item["clip_name"]
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        item["output_name"]
    )

    extract_frame(
        video_path=video_path,
        output_path=output_path,
        time_sec=item["time_sec"]
    )