import json
import numpy as np
from pathlib import Path

def json_log_training_progression(episode_results, filename="training_log.json"):
    def to_serializable(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return o  # let JSON handle other native types

    results_dict = []
    for er in episode_results:
        d = er.__dict__.copy()
        # Ensure action_probs is list-of-lists of floats
        if "action_probs" in d and d["action_probs"] is not None:
            d["action_probs"] = [[float(x) for x in row] for row in d["action_probs"]]
        results_dict.append(d)

    log_dir = Path.cwd() / "training_file_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    filepath = log_dir / filename

    with open(filepath, "w") as f:
        json.dump(results_dict, f, indent=4, default=to_serializable)
