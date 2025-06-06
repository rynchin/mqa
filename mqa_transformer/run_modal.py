from config import load_config
from pathlib import Path
import sys
import json
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from mqa_transformer.modal_app import app, benchmark_model

if __name__ == "__main__":
    size = "large"  # or "large"

    config_path = Path(__file__).parent.parent / "config.yml"

    model_cfg, bench_cfg = load_config(config_path, size=size)

    with app.run():
        result = benchmark_model.remote(
            model_config_json=json.dumps(model_cfg),
            bench_config_json=json.dumps(bench_cfg)
        )
        print(result)
