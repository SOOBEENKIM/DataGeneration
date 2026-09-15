from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np
import pandas as pd
from torch import optim

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.ctgan import Discriminator, Generator

from benchmarks.types import SequenceBatch
from generators.candidate_model_backends_v2_6 import (
    CheckpointableCTGANCandidateV26,
    CoFSeqGenCandidateV26,
    ConditionalCTGANCandidateV26,
)
from generators.conditional_ctgan import _frame_hash
from models.single_factor_components_v2_6 import (
    EpsilonPredictionCoFSeqGenV26,
    epsilon_reverse_sample_v2_6,
)


class SharedTransformerCheckpointableCTGANV26(
    CheckpointableCTGANCandidateV26
):
    """Checkpointable class model using one all-train fitted transformer."""

    def set_shared_transformer(
        self,
        transformer: DataTransformer,
        *,
        fit_frame_sha256: str,
    ) -> None:
        if not isinstance(transformer, DataTransformer):
            raise TypeError("shared transformer must be a DataTransformer")
        self._v26_shared_transformer = copy.deepcopy(transformer)
        self.v2_6_shared_transformer_fit_frame_sha256 = str(
            fit_frame_sha256
        )

    def _initialize_v2_5(
        self,
        train_data: pd.DataFrame | np.ndarray,
        discrete_columns,
    ) -> None:
        self._validate_discrete_columns(
            train_data,
            discrete_columns,
        )
        self._validate_null_data(train_data, discrete_columns)
        shared = getattr(self, "_v26_shared_transformer", None)
        if not isinstance(shared, DataTransformer):
            raise ValueError(
                "shared CTGAN transformer was not fitted from train"
            )
        self._transformer = copy.deepcopy(shared)
        transformed = self._transformer.transform(train_data)
        self._v25_train_data = transformed
        self._data_sampler = DataSampler(
            transformed,
            self._transformer.output_info_list,
            self._log_frequency,
        )
        data_dim = self._transformer.output_dimensions
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)
        self._v25_discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)
        self._v25_optimizer_g = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        self._v25_optimizer_d = optim.Adam(
            self._v25_discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )
        self.loss_values = pd.DataFrame(
            columns=[
                "Step",
                "Generator Loss",
                "Discriminator Loss",
            ]
        )
        self._v25_update = 0
        self._v25_elapsed_seconds = 0.0
        self._v25_initialized = True


class ConditionalCTGANSharedTransformerV26(
    ConditionalCTGANCandidateV26
):
    """Separate-class CTGAN with one transformer fit on all train rows."""

    checkpointable_model_class = (
        SharedTransformerCheckpointableCTGANV26
    )
    baseline_definition_version = (
        "benchmark-v2.6-single-factor-shared-transformer-v1"
    )

    def fit(
        self,
        train: SequenceBatch,
        *,
        config: Mapping[str, Any],
        seed: int,
    ) -> None:
        rows = pd.DataFrame(
            {
                "amount_log": train.x_num[
                    train.valid_mask,
                    0,
                ],
                "dt_bin": train.dt_bin[train.valid_mask],
                "receiver": train.x_cat[
                    train.valid_mask,
                    0,
                ],
            }
        )
        np.random.seed(seed)
        shared = DataTransformer()
        shared.fit(
            rows,
            discrete_columns=["dt_bin", "receiver"],
        )
        self._v26_shared_transformer = shared
        self.shared_transformer_fit_frame_sha256 = _frame_hash(rows)
        self.shared_transformer_fit_split = "train"
        self.shared_transformer_fit_count = 1
        super().fit(train, config=config, seed=seed)
        transformer_dimensions = {
            model._transformer.output_dimensions
            for model in self.models.values()
        }
        transformer_hashes = {
            model.v2_6_shared_transformer_fit_frame_sha256
            for model in self.models.values()
        }
        if (
            len(transformer_dimensions) != 1
            or transformer_hashes
            != {self.shared_transformer_fit_frame_sha256}
        ):
            raise RuntimeError(
                "separate CTGAN models did not share transformer state"
            )

    def _new_model(self, config, *, epochs, checkpointable):
        model = super()._new_model(
            config,
            epochs=epochs,
            checkpointable=checkpointable,
        )
        model.set_shared_transformer(
            self._v26_shared_transformer,
            fit_frame_sha256=(
                self.shared_transformer_fit_frame_sha256
            ),
        )
        return model


class CoFNoisePredictionCandidateV26(CoFSeqGenCandidateV26):
    """Frozen CoF architecture with epsilon amount parameterization."""

    baseline_definition_version = (
        "benchmark-v2.6-single-factor-epsilon-v1"
    )
    model_class = EpsilonPredictionCoFSeqGenV26
    sampling_function = staticmethod(epsilon_reverse_sample_v2_6)
