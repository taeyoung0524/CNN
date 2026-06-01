import json
import cv2
from pathlib import Path

with open("dataset/NurViD_annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

video_dir = Path("dataset/videos")
output_dir = Path("dataset/selected_images")
output_dir.mkdir(parents=True, exist_ok=True)

op_to_label = {
    6:  "subcutaneous_injection",
    20: "intramuscular_injection",
    45: "electrocardiogram",
    36: "oxygen_therapy",
}

target_ops = set(op_to_label.keys())
extracted = {op: 0 for op in target_ops}

for video_id, info in data.items():
    op_id = info["operationID"]
    if op_id not in target_ops:
        continue

    video_path = video_dir / f"{video_id}.mp4"
    if not video_path.exists():
        continue

    label = op_to_label[op_id]
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    for i, ann in enumerate(info["annotations"]):
        start_sec, end_sec = ann["segment"]
        mid_sec = (start_sec + end_sec) / 2  # segment 중간 지점

        frame_idx = int(mid_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        out_filename = f"{label}_{video_id}_ann{i+1}.jpg"
        cv2.imwrite(str(output_dir / out_filename), frame)
        extracted[op_id] += 1

    cap.release()
    print(f"완료: {video_id} ({label}), {len(info['annotations'])}장 추출")

print("\n최종 결과:")
for op_id, count in extracted.items():
    print(f"  {op_to_label[op_id]}: {count}장")
print(f"  총: {sum(extracted.values())}장")