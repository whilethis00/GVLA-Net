"""
Run one custom2.5 precision setting and save a small JSON artifact.
"""

import argparse
import json
from pathlib import Path
import sys

import robosuite as suite

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.robosuite_quantization_study import measure_success_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Output label, e.g. inf or 256")
    parser.add_argument("--n-bins", type=int, default=None, help="Bins per dim. Omit for continuous.")
    parser.add_argument("--n-rollouts", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    env = suite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        horizon=args.max_steps,
        reward_shaping=False,
        reward_scale=1.0,
        single_object_mode=2,
        object_type="can",
    )

    config = {
        "xy_tol": 0.015,
        "release_clearance": 0.075,
        "transport_xy_thresh": 0.01,
        "place_height": 0.068,
        "kp_place": 5.4,
    }

    success_rate = measure_success_rate(
        env,
        "pick_place_can_precision",
        args.n_bins,
        args.n_rollouts,
        args.max_steps,
        decode_mode="gvla",
        precision_xy_tol=config["xy_tol"],
        precision_release_clearance=config["release_clearance"],
        precision_transport_xy_thresh=config["transport_xy_thresh"],
        precision_place_height=config["place_height"],
        precision_kp_place=config["kp_place"],
    )
    env.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "n_bins": args.n_bins,
                "n_rollouts": args.n_rollouts,
                "max_steps": args.max_steps,
                "config": config,
                "success_rate": success_rate,
                "successes": round(success_rate * args.n_rollouts),
            },
            indent=2,
        )
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
