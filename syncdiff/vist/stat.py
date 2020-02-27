from collections import defaultdict


def dataset_stat(dataset_path: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [line.replace("\n", "").split(",")[-3:] for line in f.readlines()][1:]
    result = {
        "biaoli": defaultdict(int),
        "hanre": defaultdict(int),
        "xushi": defaultdict(int)
    }
    for s in data:
        result["biaoli"][s[0]] += 1
        result["hanre"][s[1]] += 1
        result["xushi"][s[2]] += 1
    
    return result
