import json

json_path = "dataset/NurViD_annotations.json"

# 엑셀 기준 action ID
xlsx_target_actions = {
    22: "Change Wound Dressings",
    25: "Bed Shampoo",
    30: "Defibrillation",
    35: "Oxygen Therapy",
}

# JSON 기준 actionID = 엑셀 ID + 1
target_actions = {
    xlsx_id + 1: action_name
    for xlsx_id, action_name in xlsx_target_actions.items()
}

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for target_action, action_name in target_actions.items():

    print("\n")
    print("=" * 100)
    print(f"JSON ACTION {target_action}: {action_name}")
    print("=" * 100)

    count = 0

    for video_id, info in data.items():

        for ann in info["annotations"]:

            if ann["actionID"] == target_action:

                count += 1

                print(video_id)
                print(info["url"])
                print("operationID:", info["operationID"])
                print("segment:", ann["segment"])
                print()

    print(f"총 {count}개")