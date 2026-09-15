from ctgan import TVAE

from .checkpointable_tabular_v2_5 import CheckpointableTVAE
from .conditional_ctgan import ConditionalCTGAN


class ConditionalTVAE(ConditionalCTGAN):
    name = "conditional_tvae"
    model_class = TVAE
    checkpointable_model_class = CheckpointableTVAE

    def _model_kwargs(self, config, *, epochs):
        return {
            "epochs": epochs,
            "batch_size": int(config.get("batch_size", 500)),
            "l2scale": float(config.get("weight_decay", 1e-5)),
            "verbose": False,
            "cuda": bool(config.get("cuda", False)),
        }

    def _fit_checkpointable_model(
        self,
        model,
        class_frame,
        *,
        config,
        seed,
        label,
        requested_steps,
        max_wall_seconds,
        checkpoint_interval,
        progress_callback,
        checkpoint_callback,
    ):
        return model.fit_steps(
            class_frame,
            discrete_columns=["dt_bin", "receiver"],
            requested_steps=requested_steps,
            max_wall_seconds=max_wall_seconds,
            checkpoint_interval=checkpoint_interval,
            seed=seed + label,
            learning_rate=float(config.get("lr", 1e-3)),
            progress_callback=progress_callback,
            checkpoint_callback=checkpoint_callback,
        )
