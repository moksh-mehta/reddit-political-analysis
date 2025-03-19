import json

# This script will get the total record number for the data deliverable
with open("huge_output.json") as file:
    data = json.load(file)
    total = 0
    for item in data.values():
        print(item)
        total += len(item[0])
print(total)