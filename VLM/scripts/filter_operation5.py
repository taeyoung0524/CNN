import json

target_operation_ids = {5}

json_path = "Pytorch/VLM/dataset/NurViD_annotations.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

candidates = []

for video_id, info in data.items():
    if info["operationID"] in target_operation_ids:
        candidates.append({
            "video_id": video_id,
            "operationID": info["operationID"],
            "url": info["url"],
            "duration": info["duration"],
            "resolution": info["resolution"],
            "fps": info["fps"],
            "annotations": info["annotations"]
        })

print("operationID=5 후보 영상 개수:", len(candidates)) # 60개

for item in candidates[:20]:
    print("=" * 80)
    print("video_id:", item["video_id"])
    print("url:", item["url"])
    print("duration:", item["duration"])
    print("resolution:", item["resolution"])
    print("annotations:", item["annotations"])


    interesting_actions = {5, 8, 19}

for video_id, info in data.items():
    if info["operationID"] == 5:

        for ann in info["annotations"]:
            if ann["actionID"] in interesting_actions:
                print(video_id)
                print(info["url"])
                print("action:", ann["actionID"])
                print("segment:", ann["segment"])
                print()