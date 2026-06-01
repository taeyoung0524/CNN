# 노트북에서 실행
import json

with open("dataset/NurViD_annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

op_to_label = {
    6:  "subcutaneous_injection",
    20: "intramuscular_injection",
    45: "electrocardiogram",
    36: "oxygen_therapy",
}

# 각 operationID별 등장하는 actionID 확인
for op_id, label in op_to_label.items():
    action_count = {}
    for info in data.values():
        if info["operationID"] != op_id:
            continue
        for ann in info["annotations"]:
            aid = ann["actionID"]
            action_count[aid] = action_count.get(aid, 0) + 1

    print(f"\n{label} (op={op_id}) actionID 목록:")
    for aid, cnt in sorted(action_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  actionID={aid}: {cnt}회")