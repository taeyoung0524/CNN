import json

target_operation_ids = {10}

json_path = "dataset/NurViD_annotations.json"

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

print("operationID=7 후보 영상 개수:", len(candidates)) # 60개

for item in candidates[:5]:
    print("=" * 80)
    print("video_id:", item["video_id"])
    print("url:", item["url"])
    print("duration:", item["duration"])
    print("resolution:", item["resolution"])
    print("annotations:", item["annotations"])


    interesting_actions = {102}

for video_id, info in data.items():
    if info["operationID"] == 10:

        for ann in info["annotations"]:
            if ann["actionID"] in interesting_actions:
                print(video_id)
                print(info["url"])
                print("action:", ann["actionID"])
                print("segment:", ann["segment"])
                print()


# import json

# with open("dataset/NurViD_annotations.json", "r") as f:
#     data = json.load(f)

# action_count = {}

# for video_id, info in data.items():

#     if info["operationID"] == 7:

#         for ann in info["annotations"]:

#             action_id = ann["actionID"]

#             action_count[action_id] = (
#                 action_count.get(action_id, 0) + 1
#             )

# print("operationID=7 에 등장하는 action")

# for action_id, count in sorted(
#         action_count.items(),
#         key=lambda x: x[1],
#         reverse=True
# ):
#     print(action_id, count)