import os
import json
import random


def data_partition(data_folder: str):
    data = {"inputs": [], "biaoli": [], "hanre": [], "xushi": []}

    with open(os.path.join(data_folder, "raw.csv"), "r", encoding="utf-8") as f:
        lines = [line.replace("\n", "") for line in f.readlines()][1:]
        random.shuffle(lines)
        for i in range(len(lines)):
            tokens = lines[i].split(",")
            x = [int(j) for j in tokens[:-3]]
            y = [int(j) for j in tokens[-3:]]
            data["inputs"].append(x)
            data["biaoli"].append(y[0])
            data["hanre"].append(y[1])
            data["xushi"].append(y[2]) 
    with open(os.path.join(data_folder, "data.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(data))
    print("data: {0}".format(len(data["inputs"])))
