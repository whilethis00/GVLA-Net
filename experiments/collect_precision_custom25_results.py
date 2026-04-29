"""
Collect per-setting custom2.5 precision results into the expected core JSON.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARTS_DIR = ROOT / "experiments" / "results" / "precision_custom25_200roll_parts"
OUT_PATH = ROOT / "experiments" / "results" / "precision_custom25_200roll_core.json"


def main():
    labels = ["inf", "256", "1024", "2048"]
    parts = {}
    config = None
    for label in labels:
        path = PARTS_DIR / f"{label}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text())
        parts[label] = data["success_rate"]
        config = data["config"]

    OUT_PATH.write_text(json.dumps({"config": config, "results": parts}, indent=2))
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
