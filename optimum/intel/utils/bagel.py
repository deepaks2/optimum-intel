# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from transformers import AutoConfig, PretrainedConfig


class BagelConfig(PretrainedConfig):
    """Configuration for ByteDance-Seed/BAGEL-7B-MoT."""

    model_type = "bagel"

    def __init__(
        self,
        visual_gen: bool = True,
        visual_und: bool = True,
        llm_config: Optional[Dict[str, Any]] = None,
        vit_config: Optional[Dict[str, Any]] = None,
        vae_config: Optional[Dict[str, Any]] = None,
        latent_patch_size: int = 2,
        max_latent_size: int = 32,
        vit_max_num_patch_per_side: int = 70,
        connector_act: str = "gelu_pytorch_tanh",
        interpolate_pos: bool = False,
        timestep_shift: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        *args,
        **kwargs,
    ) -> "BagelConfig":
        config_file = Path(pretrained_model_name_or_path) / "config.json"
        if config_file.is_file():
            with config_file.open(encoding="utf-8") as file:
                config_dict = json.load(file)
            if config_dict.get("model_type") == cls.model_type:
                return cls.from_dict(config_dict, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


def is_bagel_model(
    model_name_or_path: Union[str, Path],
    *,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[Union[bool, str]] = None,
    subfolder: str = "",
    local_files_only: bool = False,
) -> bool:
    """Return whether a local model directory or Hub repository identifies BAGEL."""

    config_file = Path(model_name_or_path) / "config.json"
    try:
        if not config_file.is_file():
            config_file = Path(
                hf_hub_download(
                    repo_id=str(model_name_or_path),
                    filename=f"{subfolder}/config.json" if subfolder else "config.json",
                    cache_dir=cache_dir,
                    revision=revision,
                    token=token,
                    local_files_only=local_files_only,
                )
            )
        with config_file.open(encoding="utf-8") as file:
            return json.load(file).get("model_type") == BagelConfig.model_type
    except (OSError, ValueError, HfHubHTTPError):
        return False


def register_bagel_config() -> None:
    """Register BAGEL only when the OpenVINO exporter is imported."""

    try:
        AutoConfig.register(BagelConfig.model_type, BagelConfig)
    except ValueError as error:
        if "already registered" not in str(error):
            raise
