# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""OpenVINO export and inference support for the BAGEL MoT architecture.

BAGEL is not a Transformers-native model.  Its OpenVINO representation is
therefore deliberately componentized: token embeddings, a graph-friendly MoT
decoder, packed SigLIP tokens, latent conditioning, and the VAE encoder and
decoder are exported separately.  The runtime owns the packing and denoising
orchestration.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import openvino as ov
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import GenerationConfig, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from ..utils.bagel import BagelConfig, register_bagel_config
from .bagel_reference import load_reference_classes


logger = logging.getLogger(__name__)

BAGEL_LANGUAGE_MODEL_NAME = "openvino_language_model.xml"
BAGEL_TEXT_EMBEDDINGS_NAME = "openvino_text_embeddings_model.xml"
BAGEL_VISION_MODEL_NAME = "openvino_vision_model.xml"
BAGEL_LATENT_EMBEDDINGS_NAME = "openvino_latent_embeddings_model.xml"
BAGEL_VAE_ENCODER_NAME = "openvino_vae_encoder_model.xml"
BAGEL_VAE_DECODER_NAME = "openvino_vae_decoder_model.xml"
BAGEL_METADATA_NAME = "bagel_openvino_config.json"


def _resolve_file(
    model_name_or_path: Union[str, Path],
    filename: str,
    *,
    cache_dir: str,
    revision: Optional[str],
    token,
    subfolder: str,
    local_files_only: bool,
    force_download: bool,
) -> Path:
    local_path = Path(model_name_or_path) / subfolder / filename
    if local_path.is_file():
        return local_path
    return Path(
        hf_hub_download(
            repo_id=str(model_name_or_path),
            filename=f"{subfolder}/{filename}" if subfolder else filename,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
        )
    )


def _load_bagel_components(
    model_name_or_path: Union[str, Path],
    *,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    revision: Optional[str] = "main",
    token=None,
    subfolder: str = "",
    local_files_only: bool = False,
    force_download: bool = False,
):
    """Construct BAGEL's reference modules and attach the published weights."""

    Bagel, ReferenceBagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel = (
        load_reference_classes()
    )
    from accelerate import init_empty_weights
    from modeling.autoencoder import load_ae
    from safetensors.torch import load_file

    config_path = _resolve_file(
        model_name_or_path,
        "config.json",
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        subfolder=subfolder,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    with config_path.open(encoding="utf-8") as file:
        config_dict = json.load(file)

    llm_path = _resolve_file(
        model_name_or_path,
        "llm_config.json",
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        subfolder=subfolder,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    vit_path = _resolve_file(
        model_name_or_path,
        "vit_config.json",
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        subfolder=subfolder,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    checkpoint_path = _resolve_file(
        model_name_or_path,
        "ema.safetensors",
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        subfolder=subfolder,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    ae_path = _resolve_file(
        model_name_or_path,
        "ae.safetensors",
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        subfolder=subfolder,
        local_files_only=local_files_only,
        force_download=force_download,
    )

    llm_config = Qwen2Config.from_json_file(str(llm_path))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(str(vit_path))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=str(ae_path))
    vae_model.eval()
    vae_model.reg.sample = False

    bagel_config = ReferenceBagelConfig(
        visual_gen=bool(config_dict.get("visual_gen", True)),
        visual_und=bool(config_dict.get("visual_und", True)),
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        latent_patch_size=int(config_dict.get("latent_patch_size", 2)),
        max_latent_size=int(config_dict.get("max_latent_size", 32)),
        vit_max_num_patch_per_side=int(config_dict.get("vit_max_num_patch_per_side", 70)),
        connector_act=config_dict.get("connector_act", "gelu_pytorch_tanh"),
        interpolate_pos=bool(config_dict.get("interpolate_pos", False)),
        timestep_shift=float(config_dict.get("timestep_shift", 1.0)),
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, bagel_config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    state_dict = load_file(str(checkpoint_path), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        logger.warning(
            "BAGEL checkpoint load reported missing=%d unexpected=%d tensors", len(missing), len(unexpected)
        )
    model.eval()
    return model, vae_model, bagel_config, config_dict


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, rotary
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = rotary.inv_freq
    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.to(torch.float32)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=q.dtype).unsqueeze(1)
    sin = emb.sin().to(dtype=q.dtype).unsqueeze(1)
    return (q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin)


class _BagelTextEmbeddings(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class _BagelLanguageGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.language_model.model
        self.lm_head = model.language_model.lm_head
        self.llm2vae = model.llm2vae
        self.use_moe = bool(self.decoder.use_moe)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_mask: torch.Tensor,
    ):
        hidden = inputs_embeds
        rotary = self.decoder.rotary_emb
        cos_positions = position_ids
        for layer in self.decoder.layers:
            generation_mask_expanded = generation_mask.to(hidden.dtype).unsqueeze(-1)
            residual = hidden

            und_hidden = layer.input_layernorm(hidden)
            if self.use_moe:
                gen_hidden = layer.input_layernorm_moe_gen(hidden)
                hidden_for_und = torch.where(generation_mask_expanded.bool(), gen_hidden, und_hidden)
            else:
                hidden_for_und = und_hidden

            q_und = layer.self_attn.q_proj(hidden_for_und)
            k_und = layer.self_attn.k_proj(hidden_for_und)
            v_und = layer.self_attn.v_proj(hidden_for_und)
            q_und = layer.self_attn.q_norm(q_und.view(q_und.shape[0], q_und.shape[1], -1, layer.self_attn.head_dim))
            k_und = layer.self_attn.k_norm(k_und.view(k_und.shape[0], k_und.shape[1], -1, layer.self_attn.head_dim))

            if self.use_moe:
                q_gen = layer.self_attn.q_proj_moe_gen(gen_hidden)
                k_gen = layer.self_attn.k_proj_moe_gen(gen_hidden)
                v_gen = layer.self_attn.v_proj_moe_gen(gen_hidden)
                q_gen = layer.self_attn.q_norm_moe_gen(
                    q_gen.view(q_gen.shape[0], q_gen.shape[1], -1, layer.self_attn.head_dim)
                )
                k_gen = layer.self_attn.k_norm_moe_gen(
                    k_gen.view(k_gen.shape[0], k_gen.shape[1], -1, layer.self_attn.head_dim)
                )
                q = torch.where(generation_mask_expanded.unsqueeze(-1).bool(), q_gen, q_und)
                k = torch.where(generation_mask_expanded.unsqueeze(-1).bool(), k_gen, k_und)
                v_und = v_und.view(v_und.shape[0], v_und.shape[1], -1, layer.self_attn.head_dim)
                v_gen = v_gen.view(v_gen.shape[0], v_gen.shape[1], -1, layer.self_attn.head_dim)
                v = torch.where(generation_mask_expanded.unsqueeze(-1).bool(), v_gen, v_und)
            else:
                q = q_und
                k = k_und
                v = v_und.view(v_und.shape[0], v_und.shape[1], -1, layer.self_attn.head_dim)

            q, k = _apply_rope(q.transpose(1, 2), k.transpose(1, 2), cos_positions, rotary)
            v = v.transpose(1, 2)
            groups = layer.self_attn.num_key_value_groups
            if groups != 1:
                k = k.repeat_interleave(groups, dim=1)
                v = v.repeat_interleave(groups, dim=1)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=0.0)
            attn = attn.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)

            out_und = layer.self_attn.o_proj(attn)
            if self.use_moe:
                out_gen = layer.self_attn.o_proj_moe_gen(attn)
                out = torch.where(generation_mask_expanded.bool(), out_gen, out_und)
            else:
                out = out_und
            hidden = residual + out

            residual = hidden
            und_hidden = layer.post_attention_layernorm(hidden)
            und_hidden = layer.mlp(und_hidden)
            if self.use_moe:
                gen_hidden = layer.post_attention_layernorm_moe_gen(hidden)
                gen_hidden = layer.mlp_moe_gen(gen_hidden)
                hidden = torch.where(generation_mask_expanded.bool(), gen_hidden, und_hidden)
            else:
                hidden = und_hidden
            hidden = residual + hidden

        if self.use_moe:
            hidden = torch.where(
                generation_mask_expanded.bool(),
                self.decoder.norm_moe_gen(hidden),
                self.decoder.norm(hidden),
            )
        else:
            hidden = self.decoder.norm(hidden)
        return self.lm_head(hidden), self.llm2vae(hidden)


class _BagelVisionGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        vision = model.vit_model.vision_model
        self.embeddings = vision.embeddings
        self.layers = vision.encoder.layers
        self.post_layernorm = vision.post_layernorm
        self.connector = model.connector
        self.position_embedding = model.vit_pos_embed
        self.num_heads = model.config.vit_config.num_attention_heads
        self.head_dim = model.config.vit_config.hidden_size // self.num_heads

    def forward(self, packed_pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embeddings.patch_embedding(packed_pixel_values)
        for layer in self.layers:
            residual = hidden
            x = layer.layer_norm1(hidden)
            q = layer.self_attn.q_proj(x).view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
            k = layer.self_attn.k_proj(x).view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
            v = layer.self_attn.v_proj(x).view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            x = layer.self_attn.out_proj(x.transpose(1, 2).reshape(hidden.shape))
            hidden = residual + x
            residual = hidden
            hidden = residual + layer.mlp(layer.layer_norm2(hidden))
        hidden = self.post_layernorm(hidden)
        return self.connector(hidden) + self.position_embedding(position_ids)


class _BagelLatentEmbeddings(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vae2llm = model.vae2llm
        self.time_embedder = model.time_embedder
        self.position_embedding = model.latent_pos_embed

    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, _ = latent.shape
        latent_embed = self.vae2llm(latent)
        time_embed = self.time_embedder(timesteps.reshape(-1)).reshape(batch, tokens, -1)
        pos_embed = self.position_embedding(position_ids)
        return latent_embed + time_embed + pos_embed


class _BagelVAEEncoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(pixel_values)


class _BagelVAEDecoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents)


def _save_ir(module: torch.nn.Module, example_inputs: Tuple[torch.Tensor, ...], output: Path, input_names):
    module.eval()
    converted = ov.convert_model(module, example_input=example_inputs)
    for tensor, name in zip(converted.inputs, input_names):
        tensor.get_tensor().set_names({name})
    ov.save_model(converted, output, compress_to_fp16=False)


def export_bagel(
    model_name_or_path: Union[str, Path],
    output: Union[str, Path],
    *,
    cache_dir: str = HUGGINGFACE_HUB_CACHE,
    revision: Optional[str] = "main",
    token=None,
    subfolder: str = "",
    local_files_only: bool = False,
    force_download: bool = False,
    sequence_length: int = 4096,
    device: str = "cpu",
    **kwargs,
):
    """Export BAGEL's component graphs to OpenVINO IR."""

    if device.lower() not in {"cpu", "gpu", "npu"}:
        raise ValueError(f"BAGEL export supports CPU, GPU, and NPU targets, got {device!r}.")
    if sequence_length < 1:
        raise ValueError("`sequence_length` must be positive.")

    model, vae, reference_config, config_dict = _load_bagel_components(
        model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        subfolder=subfolder,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    external_config = BagelConfig.from_dict(config_dict)
    external_config.save_pretrained(output)
    metadata = {
        "sequence_length": sequence_length,
        "vision_tokens": int(reference_config.vit_max_num_patch_per_side**2),
        "hidden_size": int(reference_config.llm_config.hidden_size),
        "latent_channels": int(reference_config.vae_config.z_channels),
        "latent_patch_size": int(reference_config.latent_patch_size),
        "latent_downsample": int(reference_config.vae_config.downsample * reference_config.latent_patch_size),
        "max_latent_tokens": int(reference_config.max_latent_size**2),
    }
    (output / BAGEL_METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    hidden_size = reference_config.llm_config.hidden_size
    latent_dim = reference_config.latent_patch_size**2 * reference_config.vae_config.z_channels
    text_ids = torch.zeros((1, sequence_length), dtype=torch.long)
    embeddings = torch.zeros((1, sequence_length, hidden_size), dtype=torch.bfloat16)
    position_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.zeros((1, 1, sequence_length, sequence_length), dtype=torch.bfloat16)
    generation_mask = torch.zeros((1, sequence_length), dtype=torch.bool)
    _save_ir(
        _BagelTextEmbeddings(model.language_model.model.embed_tokens),
        (text_ids,),
        output / BAGEL_TEXT_EMBEDDINGS_NAME,
        ["input_ids"],
    )
    _save_ir(
        _BagelLanguageGraph(model),
        (embeddings, position_ids, attention_mask, generation_mask),
        output / BAGEL_LANGUAGE_MODEL_NAME,
        ["inputs_embeds", "position_ids", "attention_mask", "generation_mask"],
    )

    vision_tokens = int(reference_config.vit_max_num_patch_per_side**2)
    patch_dim = int(reference_config.vit_config.num_channels * reference_config.vit_config.patch_size**2)
    vision_pixels = torch.zeros((1, vision_tokens, patch_dim), dtype=torch.bfloat16)
    vision_positions = torch.arange(vision_tokens, dtype=torch.long).unsqueeze(0)
    _save_ir(
        _BagelVisionGraph(model),
        (vision_pixels, vision_positions),
        output / BAGEL_VISION_MODEL_NAME,
        ["packed_pixel_values", "position_ids"],
    )

    latent_tokens = int(reference_config.max_latent_size**2)
    latent = torch.zeros((1, latent_tokens, latent_dim), dtype=torch.bfloat16)
    timestep = torch.zeros((1, latent_tokens), dtype=torch.float32)
    latent_position = torch.zeros((1, latent_tokens), dtype=torch.long)
    _save_ir(
        _BagelLatentEmbeddings(model),
        (latent, timestep, latent_position),
        output / BAGEL_LATENT_EMBEDDINGS_NAME,
        ["latent", "timesteps", "position_ids"],
    )

    vae_pixels = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    vae_latents = torch.zeros((1, reference_config.vae_config.z_channels, 32, 32), dtype=torch.float32)
    _save_ir(_BagelVAEEncoder(vae), (vae_pixels,), output / BAGEL_VAE_ENCODER_NAME, ["pixel_values"])
    _save_ir(_BagelVAEDecoder(vae), (vae_latents,), output / BAGEL_VAE_DECODER_NAME, ["latents"])
    return [
        str(output / name)
        for name in (
            BAGEL_LANGUAGE_MODEL_NAME,
            BAGEL_TEXT_EMBEDDINGS_NAME,
            BAGEL_VISION_MODEL_NAME,
            BAGEL_LATENT_EMBEDDINGS_NAME,
            BAGEL_VAE_ENCODER_NAME,
            BAGEL_VAE_DECODER_NAME,
        )
    ]


class OVBagelForConditionalGeneration:
    """OpenVINO BAGEL runtime for text/VLM inference and flow-matching generation."""

    export_feature = "image-text-to-text"
    auto_model_class = None

    def __init__(
        self,
        language_model: ov.Model,
        text_embeddings: ov.Model,
        vision_model: ov.Model,
        latent_embeddings: ov.Model,
        vae_encoder: ov.Model,
        vae_decoder: ov.Model,
        config: PretrainedConfig,
        *,
        device: str = "CPU",
        compile: bool = True,
        generation_config: Optional[GenerationConfig] = None,
        model_save_dir: Optional[Union[str, Path]] = None,
    ):
        self.config = config
        self.model_save_dir = model_save_dir
        self._device = device.upper()
        self._models = {
            "language_model": language_model,
            "text_embeddings": text_embeddings,
            "vision_model": vision_model,
            "latent_embeddings": latent_embeddings,
            "vae_encoder": vae_encoder,
            "vae_decoder": vae_decoder,
        }
        self._compiled = {}
        self.generation_config = generation_config or GenerationConfig.from_model_config(config)
        if compile:
            self.compile()

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        *,
        export: bool = False,
        device: str = "CPU",
        config: Optional[PretrainedConfig] = None,
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        token=None,
        local_files_only: bool = False,
        force_download: bool = False,
        **kwargs,
    ):
        if Path(model_id).is_dir():
            model_dir = Path(model_id)
        else:
            model_dir = (
                Path(cache_dir)
                / "optimum-intel"
                / "bagel"
                / str(model_id).replace("/", "--")
                / (revision or "main")
            )
        if export or not (model_dir / BAGEL_LANGUAGE_MODEL_NAME).is_file():
            from optimum.exporters.openvino import main_export

            main_export(
                str(model_id),
                output=model_dir,
                task="image-text-to-text",
                device=device,
                revision=revision or "main",
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
                force_download=force_download,
            )
        config = config or BagelConfig.from_pretrained(model_dir)
        components = {
            name: cls._load_component(model_dir, filename)
            for name, filename in {
                "language_model": BAGEL_LANGUAGE_MODEL_NAME,
                "text_embeddings": BAGEL_TEXT_EMBEDDINGS_NAME,
                "vision_model": BAGEL_VISION_MODEL_NAME,
                "latent_embeddings": BAGEL_LATENT_EMBEDDINGS_NAME,
                "vae_encoder": BAGEL_VAE_ENCODER_NAME,
                "vae_decoder": BAGEL_VAE_DECODER_NAME,
            }.items()
        }
        return cls(
            components.pop("language_model"),
            **components,
            config=config,
            device=device,
            model_save_dir=model_dir,
            **kwargs,
        )

    @staticmethod
    def _load_component(model_dir: Path, filename: str) -> ov.Model:
        path = model_dir / filename
        if not path.is_file():
            raise OSError(f"Missing BAGEL OpenVINO component: {path}")
        return ov.Core().read_model(path)

    def compile(self):
        core = ov.Core()
        self._compiled = {
            name: core.compile_model(model, self._device)
            for name, model in self._models.items()
        }
        return self

    def _infer(self, name: str, inputs: Dict[str, np.ndarray]):
        if name not in self._compiled:
            self.compile()
        result = self._compiled[name](inputs)
        return [np.asarray(value) for value in result.values()]

    def _metadata(self) -> Dict[str, int]:
        if self.model_save_dir is None:
            raise RuntimeError("BAGEL metadata is unavailable for an in-memory model.")
        with (Path(self.model_save_dir) / BAGEL_METADATA_NAME).open(encoding="utf-8") as file:
            return json.load(file)

    def _sequence_length(self) -> int:
        return int(self._metadata()["sequence_length"])

    def _pad_language_inputs(self, inputs_embeds, position_ids, attention_mask, generation_mask):
        max_length = self._sequence_length()
        length = inputs_embeds.shape[1]
        if length > max_length:
            raise ValueError(f"BAGEL input length {length} exceeds the exported limit of {max_length}.")
        if attention_mask is None:
            causal = np.tril(np.ones((length, length), dtype=bool))
            attention_mask = np.where(causal, 0.0, -np.inf)[None, None, :, :]
        if length == max_length:
            return inputs_embeds, position_ids, attention_mask, generation_mask, length

        padded_embeds = np.zeros(
            (inputs_embeds.shape[0], max_length, inputs_embeds.shape[2]), dtype=inputs_embeds.dtype
        )
        padded_embeds[:, :length] = inputs_embeds
        padded_positions = np.zeros((position_ids.shape[0], max_length), dtype=position_ids.dtype)
        padded_positions[:, :length] = position_ids
        padded_generation = np.zeros((generation_mask.shape[0], max_length), dtype=bool)
        padded_generation[:, :length] = generation_mask

        padded_mask = np.full((1, 1, max_length, max_length), -np.inf, dtype=attention_mask.dtype)
        padded_mask[:, :, :length, :length] = attention_mask
        return padded_embeds, padded_positions, padded_mask, padded_generation, length

    def forward(
        self, input_ids=None, inputs_embeds=None, attention_mask=None, generation_mask=None, position_ids=None
    ):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")
            input_ids = np.asarray(input_ids, dtype=np.int64)
            input_length = input_ids.shape[1] if input_ids.ndim == 2 else input_ids.shape[0]
            max_length = self._sequence_length()
            if input_length > max_length:
                raise ValueError(f"BAGEL input length {input_length} exceeds the exported limit of {max_length}.")
            padded_ids = np.zeros((1, max_length), dtype=np.int64)
            padded_ids[:, :input_length] = input_ids.reshape(1, -1)
            inputs_embeds = self._infer("text_embeddings", {"input_ids": padded_ids})[0][:, :input_length]
        inputs_embeds = np.asarray(inputs_embeds)
        sequence_length = inputs_embeds.shape[1]
        if position_ids is None:
            position_ids = np.arange(sequence_length, dtype=np.int64)[None, :]
        if generation_mask is None:
            generation_mask = np.zeros((1, sequence_length), dtype=bool)
        inputs_embeds, position_ids, attention_mask, generation_mask, length = self._pad_language_inputs(
            inputs_embeds,
            np.asarray(position_ids, dtype=np.int64),
            None if attention_mask is None else np.asarray(attention_mask),
            np.asarray(generation_mask),
        )
        outputs = self._infer(
            "language_model",
            {
                "inputs_embeds": inputs_embeds,
                "position_ids": np.asarray(position_ids, dtype=np.int64),
                "attention_mask": np.asarray(attention_mask),
                "generation_mask": np.asarray(generation_mask),
            },
        )
        return (
            CausalLMOutput(logits=torch.from_numpy(outputs[0][:, :length]), hidden_states=None),
            torch.from_numpy(outputs[1][:, :length]),
        )

    def generate(
        self, input_ids, max_new_tokens: int = 128, temperature: float = 1.0, do_sample: bool = False, **kwargs
    ):
        input_ids = np.asarray(input_ids, dtype=np.int64)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        generated = input_ids.copy()
        for _ in range(max_new_tokens):
            if generated.shape[1] >= self._sequence_length():
                raise ValueError("BAGEL generation reached the exported sequence length.")
            logits = self.forward(input_ids=generated)[0].logits[:, -1]
            if do_sample:
                probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).numpy()
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True).numpy()
            generated = np.concatenate((generated, next_token), axis=1)
        return torch.from_numpy(generated)

    def encode_image(self, packed_pixel_values, position_ids=None):
        pixels = np.asarray(packed_pixel_values)
        max_tokens = int(self._metadata()["vision_tokens"])
        tokens = pixels.shape[1]
        if tokens > max_tokens:
            raise ValueError(f"BAGEL image has {tokens} patches; the exported limit is {max_tokens}.")
        if tokens < max_tokens:
            padded = np.zeros((pixels.shape[0], max_tokens, pixels.shape[2]), dtype=pixels.dtype)
            padded[:, :tokens] = pixels
            pixels = padded
        if position_ids is None:
            position_ids = np.arange(tokens, dtype=np.int64)[None, :]
        if position_ids.shape[1] < max_tokens:
            padded_positions = np.zeros((position_ids.shape[0], max_tokens), dtype=position_ids.dtype)
            padded_positions[:, : position_ids.shape[1]] = position_ids
            position_ids = padded_positions
        return torch.from_numpy(
            self._infer(
                "vision_model",
                {
                    "packed_pixel_values": pixels,
                    "position_ids": np.asarray(position_ids, dtype=np.int64),
                },
            )[0]
        )

    def latent_embeddings(self, latent, timesteps, position_ids):
        max_tokens = int(self._metadata()["max_latent_tokens"])
        tokens = latent.shape[1]
        if tokens > max_tokens:
            raise ValueError(f"BAGEL latent sequence has {tokens} tokens; the exported limit is {max_tokens}.")
        if tokens < max_tokens:
            padded_latent = np.zeros((latent.shape[0], max_tokens, latent.shape[2]), dtype=np.asarray(latent).dtype)
            padded_latent[:, :tokens] = latent
            latent = padded_latent
            padded_timesteps = np.zeros((timesteps.shape[0], max_tokens), dtype=np.asarray(timesteps).dtype)
            padded_timesteps[:, :tokens] = timesteps
            timesteps = padded_timesteps
            padded_positions = np.zeros((position_ids.shape[0], max_tokens), dtype=np.asarray(position_ids).dtype)
            padded_positions[:, :tokens] = position_ids
            position_ids = padded_positions
        return torch.from_numpy(
            self._infer(
                "latent_embeddings",
                {
                    "latent": np.asarray(latent),
                    "timesteps": np.asarray(timesteps, dtype=np.float32),
                    "position_ids": np.asarray(position_ids, dtype=np.int64),
                },
            )[0][:, :tokens]
        )

    def decode_latents(self, latents):
        image = self._infer("vae_decoder", {"latents": np.asarray(latents)})[0]
        return torch.from_numpy(image)

    def encode_pixels(self, pixels):
        return torch.from_numpy(self._infer("vae_encoder", {"pixel_values": np.asarray(pixels)})[0])

    def _flow_attention_mask(self, context_length: int, latent_length: int) -> np.ndarray:
        sequence_length = context_length + latent_length
        mask = np.full((1, 1, sequence_length, sequence_length), -np.inf, dtype=np.float32)
        mask[:, :, :context_length, :context_length] = np.where(
            np.tril(np.ones((context_length, context_length), dtype=bool)), 0.0, -np.inf
        )
        mask[:, :, context_length:, :sequence_length] = 0.0
        return mask

    def generate_image(
        self,
        input_ids,
        *,
        image_size=(256, 256),
        num_timesteps: int = 24,
        seed: Optional[int] = None,
        negative_input_ids=None,
        guidance_scale: float = 1.0,
        initial_latent: Optional[torch.Tensor] = None,
    ):
        """Generate an image using the exported flow graph.

        ``input_ids`` must already contain the BAGEL prompt template.  Image
        preprocessing and chat-template construction remain tokenizer-specific.
        """

        height, width = image_size
        metadata = self._metadata()
        downsample = int(metadata["latent_downsample"])
        patch_size = int(metadata["latent_patch_size"])
        latent_channels = int(metadata["latent_channels"])
        latent_height = height // downsample
        latent_width = width // downsample
        latent_tokens = (latent_height // patch_size) * (latent_width // patch_size)
        if latent_height % patch_size or latent_width % patch_size:
            raise ValueError("BAGEL image dimensions must produce whole latent patches.")
        if latent_height * latent_width != 32 * 32:
            raise ValueError("The exported VAE graph currently supports 256x256 BAGEL images.")
        if num_timesteps < 2:
            raise ValueError("`num_timesteps` must be at least 2.")

        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        latent_dim = latent_channels * patch_size**2
        if initial_latent is None:
            x_t = torch.randn((1, latent_tokens, latent_dim), generator=generator, dtype=torch.float32)
        else:
            x_t = torch.as_tensor(initial_latent, dtype=torch.float32).clone()
            if tuple(x_t.shape) != (1, latent_tokens, latent_dim):
                raise ValueError(f"`initial_latent` must have shape {(1, latent_tokens, latent_dim)}.")
        text_ids = np.asarray(input_ids, dtype=np.int64).reshape(1, -1)
        if text_ids.shape[1] > self._sequence_length():
            raise ValueError("The BAGEL prompt exceeds the exported sequence length.")
        context_embeds = self._infer(
            "text_embeddings",
            {
                "input_ids": np.pad(
                    text_ids, ((0, 0), (0, self._sequence_length() - text_ids.shape[1]))
                )
            },
        )[0][:, : text_ids.shape[1]]
        negative_context = None
        if negative_input_ids is not None:
            negative_ids = np.asarray(negative_input_ids, dtype=np.int64).reshape(1, -1)
            if guidance_scale != 1.0:
                if negative_ids.shape[1] > self._sequence_length():
                    raise ValueError("The negative BAGEL prompt exceeds the exported sequence length.")
                negative_context = self._infer(
                    "text_embeddings",
                    {
                        "input_ids": np.pad(
                            negative_ids, ((0, 0), (0, self._sequence_length() - negative_ids.shape[1]))
                        )
                    },
                )[0][:, : negative_ids.shape[1]]

        latent_positions = np.arange(latent_tokens, dtype=np.int64)[None, :]
        timesteps = torch.linspace(1.0, 0.0, num_timesteps)
        for index in range(num_timesteps - 1):
            timestep = timesteps[index].expand(1, latent_tokens).numpy()
            latent_embed = self.latent_embeddings(
                x_t.numpy(),
                timestep,
                np.broadcast_to(latent_positions, (1, latent_tokens)),
            ).numpy()
            sequence = np.concatenate((context_embeds, latent_embed), axis=1)
            positions = np.concatenate(
                (
                    np.arange(context_embeds.shape[1], dtype=np.int64)[None, :],
                    np.full((1, latent_tokens), context_embeds.shape[1], dtype=np.int64),
                ),
                axis=1,
            )
            generation_mask = np.concatenate(
                (
                    np.zeros((1, context_embeds.shape[1]), dtype=bool),
                    np.ones((1, latent_tokens), dtype=bool),
                ),
                axis=1,
            )
            velocity = self.forward(
                inputs_embeds=sequence,
                position_ids=positions,
                attention_mask=self._flow_attention_mask(context_embeds.shape[1], latent_tokens),
                generation_mask=generation_mask,
            )[1][:, -latent_tokens:].float()
            if negative_context is not None:
                negative_sequence = np.concatenate((negative_context, latent_embed), axis=1)
                negative_positions = np.concatenate(
                    (
                        np.arange(negative_context.shape[1], dtype=np.int64)[None, :],
                        np.full((1, latent_tokens), negative_context.shape[1], dtype=np.int64),
                    ),
                    axis=1,
                )
                negative_mask = np.concatenate(
                    (
                        np.zeros((1, negative_context.shape[1]), dtype=bool),
                        np.ones((1, latent_tokens), dtype=bool),
                    ),
                    axis=1,
                )
                unconditional = self.forward(
                    inputs_embeds=negative_sequence,
                    position_ids=negative_positions,
                    attention_mask=self._flow_attention_mask(negative_context.shape[1], latent_tokens),
                    generation_mask=negative_mask,
                )[1][:, -latent_tokens:].float()
                velocity = unconditional + guidance_scale * (velocity - unconditional)
            delta = timesteps[index] - timesteps[index + 1]
            x_t = x_t - velocity * delta

        latent = x_t.reshape(
            1, latent_height // patch_size, latent_width // patch_size, patch_size, patch_size, latent_channels
        )
        latent = latent.permute(0, 5, 1, 3, 2, 4).reshape(1, latent_channels, latent_height, latent_width)
        return self.decode_latents(latent)

    def edit_image(
        self,
        input_pixels,
        input_ids,
        *,
        num_timesteps: int = 24,
        seed: Optional[int] = None,
        noise_strength: float = 0.1,
        **kwargs,
    ):
        """Run the same flow graph starting from a VAE-encoded source image."""

        encoded = self.encode_pixels(input_pixels)
        height, width = input_pixels.shape[-2:]
        metadata = self._metadata()
        patch_size = int(metadata["latent_patch_size"])
        latent_channels = int(metadata["latent_channels"])
        downsample = int(metadata["latent_downsample"])
        latent_height, latent_width = height // downsample, width // downsample
        latent = encoded[:, :, :latent_height, :latent_width]
        latent = latent.reshape(
            1, latent_channels, latent_height // patch_size, patch_size, latent_width // patch_size, patch_size
        )
        latent = latent.permute(0, 2, 4, 3, 5, 1).reshape(
            1, (latent_height // patch_size) * (latent_width // patch_size), latent_channels * patch_size**2
        )
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        initial_latent = latent + noise_strength * torch.randn(latent.shape, generator=generator)
        return self.generate_image(
            input_ids,
            image_size=(height, width),
            num_timesteps=num_timesteps,
            seed=seed,
            initial_latent=initial_latent,
            **kwargs,
        )

    def save_pretrained(self, save_directory: Union[str, Path]):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_directory)
        for name, model in self._models.items():
            filename = {
                "language_model": BAGEL_LANGUAGE_MODEL_NAME,
                "text_embeddings": BAGEL_TEXT_EMBEDDINGS_NAME,
                "vision_model": BAGEL_VISION_MODEL_NAME,
                "latent_embeddings": BAGEL_LATENT_EMBEDDINGS_NAME,
                "vae_encoder": BAGEL_VAE_ENCODER_NAME,
                "vae_decoder": BAGEL_VAE_DECODER_NAME,
            }[name]
            ov.save_model(model, save_directory / filename, compress_to_fp16=False)
        if self.model_save_dir and Path(self.model_save_dir).joinpath(BAGEL_METADATA_NAME).is_file():
            (save_directory / BAGEL_METADATA_NAME).write_text(
                Path(self.model_save_dir, BAGEL_METADATA_NAME).read_text(encoding="utf-8"),
                encoding="utf-8",
            )


register_bagel_config()
try:
    from transformers import AutoModelForImageTextToText

    OVBagelForConditionalGeneration.auto_model_class = AutoModelForImageTextToText
    AutoModelForImageTextToText.register(BagelConfig, OVBagelForConditionalGeneration)
except (AttributeError, ValueError):
    pass
