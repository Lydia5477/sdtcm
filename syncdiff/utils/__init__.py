def load_opt(opt_path: str):
    import json
    with open(opt_path, "r", encoding="utf-8") as f:
        text = f.read()

    return json.loads(text)


def combine_hps(tune_latent_size: bool):
    learning_rates = [0.00001, 0.0001, 0.001, 0.01]
    batch_sizes = [2, 4, 8, 16, 32, 64, 128]
    latent_sizes = [16, 32, 64, 128]

    hps = []
    for lr in learning_rates:
        for bs in batch_sizes:
            if tune_latent_size:
                for ls in latent_sizes:
                    hps.append({
                        "learning_rate": lr,
                        "batch_size": bs,
                        "latent_size": ls
                    })
            else:
                hps.append({
                    "learning_rate": lr,
                    "batch_size": bs
                })
    
    return hps
