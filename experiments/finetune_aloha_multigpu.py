"""
Local multi-GPU wrapper for Octo ALOHA finetuning.

This mirrors `third_party/octo/examples/02_finetune_new_observation_action.py`,
but actually shards batches across local devices with `jax.pmap`.
"""

from absl import app, flags, logging
import flax
import jax
from jax import lax
import numpy as np
import optax
import os
import grp
import pwd
import tensorflow as tf
from functools import partial

from octo.data.dataset import make_single_dataset
from octo.model.components.action_heads import L1ActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)


FLAGS = flags.FLAGS

_ORIG_GETPWUID = pwd.getpwuid
_ORIG_GETGRGID = grp.getgrgid


def _safe_getpwuid(uid: int):
    try:
        return _ORIG_GETPWUID(uid)
    except KeyError:
        fallback_name = os.environ.get("USER") or os.environ.get("LOGNAME") or str(uid)
        return type(
            "PwdEntry",
            (),
            {
                "pw_name": fallback_name,
                "pw_passwd": "x",
                "pw_uid": uid,
                "pw_gid": os.getgid(),
                "pw_gecos": fallback_name,
                "pw_dir": os.path.expanduser("~"),
                "pw_shell": os.environ.get("SHELL", "/bin/sh"),
            },
        )()


def _safe_getgrgid(gid: int):
    try:
        return _ORIG_GETGRGID(gid)
    except KeyError:
        fallback_name = os.environ.get("GROUP") or str(gid)
        return type(
            "GrpEntry",
            (),
            {
                "gr_name": fallback_name,
                "gr_passwd": "x",
                "gr_gid": gid,
                "gr_mem": [],
            },
        )()


pwd.getpwuid = _safe_getpwuid
grp.getgrgid = _safe_getgrgid

flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained Octo checkpoint directory."
)
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 128, "Global batch size for finetuning.")
flags.DEFINE_integer("steps", 5000, "Number of finetuning steps.")
flags.DEFINE_integer("save_interval", 1000, "Checkpoint save interval.")
flags.DEFINE_integer(
    "action_horizon",
    50,
    "Action chunk horizon used for both dataset targets and the action head.",
)
flags.DEFINE_integer(
    "window_size",
    1,
    "Observation history length used for both dataset chunking and the model input.",
)
flags.DEFINE_integer(
    "num_devices",
    1,
    "Number of local CUDA devices to use. Use 1 for the stable path; 2 enables experimental pmap.",
)
flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)


def process_batch(batch, text_processor):
    batch = process_text(batch, text_processor)
    if "dataset_name" in batch:
        del batch["dataset_name"]
    return batch


def shard_batch(batch, n_devices: int):
    def _reshape(x):
        x = np.asarray(x)
        if x.shape[0] % n_devices != 0:
            raise ValueError(
                f"Batch leading dim {x.shape[0]} is not divisible by n_devices={n_devices}."
            )
        per_device = x.shape[0] // n_devices
        return x.reshape((n_devices, per_device) + x.shape[1:])

    return jax.tree_map(_reshape, batch)


def unreplicate_pytree(tree):
    return jax.tree_map(lambda x: np.array(jax.device_get(x[0])), tree)


def host_model_from_replicated(model: OctoModel) -> OctoModel:
    return model.replace(
        params=unreplicate_pytree(model.params),
        example_batch=unreplicate_pytree(model.example_batch),
        dataset_statistics=unreplicate_pytree(model.dataset_statistics)
        if model.dataset_statistics is not None
        else None,
    )


def main(_):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    local_devices = jax.local_devices()
    available_devices = len(local_devices)
    if available_devices < 1:
        raise RuntimeError("No JAX devices found.")
    if FLAGS.num_devices < 1:
        raise ValueError("--num_devices must be >= 1.")
    if FLAGS.num_devices > available_devices:
        raise ValueError(
            f"Requested num_devices={FLAGS.num_devices}, but only {available_devices} local devices are visible."
        )
    n_devices = FLAGS.num_devices
    local_devices = local_devices[:n_devices]
    if FLAGS.batch_size % n_devices != 0:
        raise ValueError(
            f"Global batch size {FLAGS.batch_size} must be divisible by local device count {n_devices}."
        )

    initialize_compilation_cache()
    tf.config.set_visible_devices([], "GPU")

    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    logging.info("Loading finetuning dataset...")
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="aloha_sim_cube_scripted_dataset",
            data_dir=FLAGS.data_dir,
            image_obs_keys={"primary": "top"},
            proprio_obs_key="state",
            language_key="language_instruction",
        ),
        traj_transform_kwargs=dict(
            window_size=FLAGS.window_size,
            action_horizon=FLAGS.action_horizon,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
        ),
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(10000)
        .batch(FLAGS.batch_size)
        .iterator()
    )

    text_processor = pretrained_model.text_processor
    example_batch = process_batch(next(train_data_iter), text_processor)

    config = pretrained_model.config
    del config["model"]["observation_tokenizers"]["wrist"]
    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-2.0,
        high=2.0,
        obs_keys=["proprio"],
    )
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        action_horizon=FLAGS.action_horizon,
        action_dim=14,
        readout_key="readout_action",
    )

    logging.info("Updating model for new observation & action space...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = list(model.config["optimizer"]["frozen_keys"])
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)

    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    def loss_fn(params, batch, rng):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=True,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=True,
        )
        return action_loss, action_metrics

    if n_devices == 1:
        train_state = jax.device_put(train_state, local_devices[0])

        @jax.jit
        def train_step(state, batch):
            rng, dropout_rng = jax.random.split(state.rng)
            (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.model.params, batch, dropout_rng
            )
            new_state = state.apply_gradients(grads=grads, rng=rng)
            return new_state, loss, info

        logging.info(
            "Starting single-GPU finetuning: devices=%d, global_batch=%d, window_size=%d, action_horizon=%d",
            n_devices,
            FLAGS.batch_size,
            FLAGS.window_size,
            FLAGS.action_horizon,
        )
    else:
        train_state = jax.device_put_replicated(train_state, local_devices)

        @partial(jax.pmap, axis_name="devices")
        def train_step(state, batch):
            rng, dropout_rng = jax.random.split(state.rng)
            (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.model.params, batch, dropout_rng
            )
            grads = lax.pmean(grads, axis_name="devices")
            info = lax.pmean(info, axis_name="devices")
            loss = lax.pmean(loss, axis_name="devices")
            new_state = state.apply_gradients(grads=grads, rng=rng)
            return new_state, loss, info

        logging.info(
            "Starting multi-GPU finetuning: devices=%d, global_batch=%d, per_device_batch=%d, window_size=%d, action_horizon=%d",
            n_devices,
            FLAGS.batch_size,
            FLAGS.batch_size // n_devices,
            FLAGS.window_size,
            FLAGS.action_horizon,
        )

    for i in range(FLAGS.steps):
        batch = process_batch(next(train_data_iter), text_processor)
        if n_devices > 1:
            batch = shard_batch(batch, n_devices)
        train_state, loss, update_info = train_step(train_state, batch)

        if (i + 1) % 100 == 0:
            if n_devices == 1:
                host_loss = float(np.array(jax.device_get(loss)))
                host_info = jax.tree_map(lambda x: np.array(jax.device_get(x)), update_info)
            else:
                host_loss = float(np.array(jax.device_get(loss[0])))
                host_info = jax.tree_map(
                    lambda x: np.array(jax.device_get(x[0])), update_info
                )
            host_info = flax.traverse_util.flatten_dict({"training": host_info}, sep="/")
            metric_preview = ", ".join(
                f"{k}={float(v):.6f}" for k, v in list(host_info.items())[:4]
            )
            logging.info("step=%d loss=%.6f %s", i + 1, host_loss, metric_preview)

        if (i + 1) % FLAGS.save_interval == 0:
            host_model = (
                train_state.model if n_devices == 1 else host_model_from_replicated(train_state.model)
            )
            host_model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir)

    host_model = train_state.model if n_devices == 1 else host_model_from_replicated(train_state.model)
    host_model.save_pretrained(step=FLAGS.steps - 1, checkpoint_path=FLAGS.save_dir)
    logging.info("Finetune complete. Checkpoints saved to %s", FLAGS.save_dir)


if __name__ == "__main__":
    app.run(main)
