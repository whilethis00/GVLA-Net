"""
Generate a compact qualitative figure for pick_place_can_precision (custom2.5).

Output:
  [Success | Success | Failure | Failure]
with minimal labels and no UI elements.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import robosuite as suite

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.robosuite_quantization_study import (
    make_policy,
    decode_action,
    is_success,
)


OUT_DIR = ROOT / "experiments" / "results" / "qualitative_precision_custom25"
CAMERA_NAME = "agentview"
IMG_SIZE = 512
CROP_BOX = (272, 132, 506, 364)  # cleaner crop around the target tray in agentview

CFG = {
    "xy_tol": 0.015,
    "release_clearance": 0.075,
    "transport_xy_thresh": 0.01,
    "place_height": 0.068,
    "kp_place": 5.4,
}


def make_env(max_steps):
    return suite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        ignore_done=True,
        horizon=max_steps,
        reward_shaping=False,
        reward_scale=1.0,
        single_object_mode=2,
        object_type="can",
    )


def render_crop(env):
    image = env.sim.render(width=IMG_SIZE, height=IMG_SIZE, camera_name=CAMERA_NAME)[::-1]
    pil = Image.fromarray(image)
    return pil.crop(CROP_BOX)


def rollout(env, seed, n_bins, max_steps=400):
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    policy = make_policy(
        "pick_place_can_precision",
        precision_transport_xy_thresh=CFG["transport_xy_thresh"],
        precision_place_height=CFG["place_height"],
        precision_kp_place=CFG["kp_place"],
    )
    policy.reset()
    obs = env.reset()
    reward_accum = 0.0
    prev_action = np.zeros(7, dtype=np.float32)

    done = False
    for _ in range(max_steps):
        action = policy.act(obs, env)
        action = decode_action(action, n_bins, "gvla", 256, 0.0, rng)
        obs, reward, done, _ = env.step(action)
        reward_accum += reward
        prev_action = action
        if done or is_success(
            env,
            obs,
            reward_accum,
            "pick_place_can_precision",
            policy,
            precision_xy_tol=CFG["xy_tol"],
            precision_release_clearance=CFG["release_clearance"],
        ):
            break

    # Let the object settle and the gripper retreat for a cleaner final frame.
    settle_action = np.zeros(7, dtype=np.float32)
    settle_action[2] = 0.75
    settle_action[6] = -1.0
    for _ in range(28):
        obs, reward, done, _ = env.step(settle_action)
        reward_accum += reward

    obj_pos = np.array(obs["Can_pos"])
    target_bin = np.array(env.target_bin_placements[env.object_id])
    eef_pos = np.array(obs["robot0_eef_pos"])
    xy_err = float(np.linalg.norm(obj_pos[:2] - target_bin[:2]))
    eef_clear = float(eef_pos[2] - obj_pos[2])
    success = is_success(
        env,
        obs,
        reward_accum,
        "pick_place_can_precision",
        policy,
        precision_xy_tol=CFG["xy_tol"],
        precision_release_clearance=CFG["release_clearance"],
    )
    return {
        "seed": seed,
        "n_bins": n_bins,
        "success": bool(success),
        "xy_err": xy_err,
        "eef_clear": eef_clear,
        "obj_pos": obj_pos.tolist(),
        "target_bin": target_bin.tolist(),
        "phase": policy.phase,
        "frame": render_crop(env),
    }


def choose_examples():
    env = make_env(max_steps=400)
    success_candidates = []
    failure_candidates = []

    # Success examples from a high-resolution setting.
    for seed in range(24):
        result = rollout(env, seed, 2048)
        if result["success"]:
            success_candidates.append(result)
        if (seed + 1) % 6 == 0:
            print("success sweep:", seed + 1, "seeds,", len(success_candidates), "candidates")
        if len(success_candidates) >= 6:
            break

    # Failure examples from a still-high-resolution setting, but with slight
    # final misalignment. This keeps the visual gap subtle instead of showing
    # catastrophic misses from extremely coarse quantization.
    for seed in range(64):
        result = rollout(env, seed, 1024)
        near_target = result["xy_err"] < 0.05
        slight_misalignment = result["xy_err"] > CFG["xy_tol"] * 1.05
        released = result["eef_clear"] > 0.04
        if (not result["success"]) and near_target and slight_misalignment and released:
            failure_candidates.append(result)
        if (seed + 1) % 8 == 0:
            print("failure sweep:", seed + 1, "seeds,", len(failure_candidates), "candidates")
        if len(failure_candidates) >= 6:
            break

    env.close()

    if len(success_candidates) < 2:
        raise RuntimeError("Could not find enough success examples.")
    if len(failure_candidates) < 2:
        raise RuntimeError("Could not find enough near-miss failure examples.")

    success_candidates.sort(key=lambda x: x["xy_err"])
    failure_candidates.sort(key=lambda x: x["xy_err"])

    # Pick two distinct examples from each side.
    chosen_success = [success_candidates[0], success_candidates[min(2, len(success_candidates) - 1)]]
    chosen_failure = [failure_candidates[0], failure_candidates[min(2, len(failure_candidates) - 1)]]
    return chosen_success + chosen_failure


def add_label(image, text):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    pad_x = 10
    pad_y = 8
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.rounded_rectangle((8, 8, 8 + w + 2 * pad_x, 8 + h + 2 * pad_y), radius=10, fill=(255, 255, 255))
    draw.text((8 + pad_x, 8 + pad_y), text, fill=(0, 0, 0), font=font)


def build_strip(examples, with_labels=True):
    tiles = []
    metadata = []
    for idx, ex in enumerate(examples):
        img = ex["frame"].copy()
        if with_labels:
            label = "success" if idx < 2 else "failure"
            add_label(img, label)
        tiles.append(img)
        meta = dict(ex)
        meta.pop("frame", None)
        metadata.append(meta)

    width = sum(img.width for img in tiles)
    height = max(img.height for img in tiles)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    cursor = 0
    for img in tiles:
        canvas.paste(img, (cursor, 0))
        cursor += img.width
    return canvas, metadata


def add_paper_margin(image, caption=None):
    top = 26
    bottom = 26 if caption else 18
    side = 18
    canvas = Image.new("RGB", (image.width + side * 2, image.height + top + bottom), (255, 255, 255))
    canvas.paste(image, (side, top))
    if caption:
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), caption, font=font)
        text_w = bbox[2] - bbox[0]
        x = (canvas.width - text_w) // 2
        y = image.height + top + 4
        draw.text((x, y), caption, fill=(25, 25, 25), font=font)
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    examples = choose_examples()
    strip, metadata = build_strip(examples, with_labels=True)
    clean_strip, _ = build_strip(examples, with_labels=False)

    out_path = OUT_DIR / "precision_success_failure_strip.png"
    clean_path = OUT_DIR / "precision_success_failure_strip_nolabel.png"
    paper_path = OUT_DIR / "precision_success_failure_strip_paper.png"

    strip.save(out_path)
    clean_strip.save(clean_path)
    add_paper_margin(
        clean_strip,
        caption="Precision-sensitive placement: small spatial deviations determine success or failure.",
    ).save(paper_path)
    (OUT_DIR / "selection_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Saved → %s" % out_path)
    print("Saved → %s" % clean_path)
    print("Saved → %s" % paper_path)
    print("Saved → %s" % (OUT_DIR / "selection_metadata.json"))


if __name__ == "__main__":
    main()
