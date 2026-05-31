import os
import json
import subprocess
from tqdm import tqdm

# 다운로드 성공한 영상만 넣기
TARGET_VIDEOS = [
    "-1mH9wYWd5w",
    "Peyw-eKZJOI",
    "oRSDFuBCia0",
    "F88btEbzdw0"
]

def cut_video_segment(input_path, output_path, start, end):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    subprocess.run([
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path
    ])

def extract_segments(annotation_path, video_dir, output_dir):
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for video_id in tqdm(TARGET_VIDEOS, desc="Cutting segments"):
        if video_id not in data:
            print(f"[WARN] {video_id} not found in annotation")
            continue

        video_path = os.path.join(video_dir, f"{video_id}.mp4")

        if not os.path.exists(video_path):
            print(f"[WARN] Video file not found: {video_path}")
            continue

        info = data[video_id]
        operation_id = info["operationID"]

        for idx, ann in enumerate(info["annotations"]):
            action_id = ann["actionID"]
            start, end = ann["segment"]

            output_filename = (
                f"{video_id}_op{operation_id}_action{action_id}_"
                f"{idx}_{start:.2f}_{end:.2f}.mp4"
            )

            output_path = os.path.join(output_dir, video_id, output_filename)

            cut_video_segment(
                input_path=video_path,
                output_path=output_path,
                start=start,
                end=end
            )

if __name__ == "__main__":
    annotation_path = "./dataset/NurViD_annotations.json"
    video_dir = "./dataset/original_video"
    output_dir = "./dataset/segments"

    extract_segments(annotation_path, video_dir, output_dir)

    print("Done. Segments saved to:", output_dir)