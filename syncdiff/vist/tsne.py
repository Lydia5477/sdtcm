import numpy as np
from sklearn.manifold import TSNE
import json
import os
import matplotlib.pyplot as plt


def visualize_raw_data(data_path: str, saved_folder: str):
    if not os.path.isdir(saved_folder):
        os.mkdir(saved_folder)
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.loads(f.read())

    font = {
        "color": "black", 
        "size": 13, 
        "family" : "serif"
    }
    plt.style.use("seaborn")
    plt.figure(figsize=(11, 6))
    sds = ["biaoli", "hanre", "xushi"]
    cs = {-1: "red", 0: "green", 1: "orange"}
    for idx in range(len(sds)):
        sd = sds[idx]
        plt.subplot(1, 3, idx + 1)
        data_map = {-1: [], 0: [], 1: []}
        for i in range(len(data["inputs"])):
            data_map[data[sd][i]].append(data["inputs"][i])
        for t, d in data_map.items():
            model = TSNE(n_components=2, random_state=0)
            np.set_printoptions(suppress=True)
            x_tsne = model.fit_transform(np.array(d))
            plt.scatter(
                x_tsne[:, 0], x_tsne[:, 1], c=cs[t], alpha=0.6, cmap=plt.cm.get_cmap('rainbow', 10))
            plt.clim(-0.5, 9.5)
        cbar = plt.colorbar(ticks=range(10)) 
        cbar.set_label(label=sd, fontdict=font)
    plt.savefig(os.path.join(saved_folder, "data_tsne.pdf"))
    plt.savefig(os.path.join(saved_folder, "data_tsne.png"))
    plt.show()
