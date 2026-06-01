# 노트북에서 실행해줘
import json

with open("dataset/NurViD_annotations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for op_id in [6, 45, 20, 36]:
    count = sum(1 for v in data.values() if v["operationID"] == op_id)
    print(f"operationID={op_id}: {count}개 영상")