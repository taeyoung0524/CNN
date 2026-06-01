import json
import subprocess
from pathlib import Path

with open("dataset/NurViD_annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 대상 operationID
target_ops = {6, 20, 45, 36}

# operationID별 최대 다운로드 수
max_per_op = {
    6: 10,   # Subcutaneous Injection
    20: 10,  # Intramuscular Injection
    45: 8,   # Electrocardiogram (전부)
    36: 10,  # Oxygen Therapy
}

output_dir = Path("dataset/videos")
output_dir.mkdir(parents=True, exist_ok=True)

op_count = {op: 0 for op in target_ops}

for video_id, info in data.items():
    op_id = info["operationID"]
    if op_id not in target_ops:
        continue
    if op_count[op_id] >= max_per_op[op_id]:
        continue

    url = info["url"]
    out_path = output_dir / f"{video_id}.mp4"

    if out_path.exists():
        print(f"이미 존재: {video_id}")
        op_count[op_id] += 1
        continue

    print(f"다운로드 중: {video_id} (op={op_id})")
    result = subprocess.run([
        "yt-dlp",
        "-f", "mp4",
        "-o", str(out_path),
        "--quiet",
        url
    ])

    if result.returncode == 0:
        op_count[op_id] += 1
        print(f"완료: {video_id} | op={op_id} ({op_count[op_id]}/{max_per_op[op_id]})")
    else:
        print(f"실패: {video_id}")

print("\n최종 다운로드 수:")
for op_id, count in op_count.items():
    print(f"  op={op_id}: {count}개")