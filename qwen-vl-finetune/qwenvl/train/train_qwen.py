# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import json
import torch
import importlib.metadata

os.environ.setdefault("USE_HUB_KERNELS", "NO")

_orig_importlib_version = importlib.metadata.version


def _hide_broken_kernels_package(package_name):
    if package_name == "kernels":
        raise importlib.metadata.PackageNotFoundError(package_name)
    return _orig_importlib_version(package_name)


importlib.metadata.version = _hide_broken_kernels_package

import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.model import Qwen3VLSegForConditionalGeneration
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.train.seg_trainer import JsonlLogCallback, TongueSegTrainer, git_commit
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def resolve_attn_implementation(attn_implementation):
    if attn_implementation != "auto":
        return attn_implementation
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def configure_seg_defaults(training_args):
    if training_args.logging_steps == 500:
        training_args.logging_steps = 10
    if training_args.save_steps == 500:
        training_args.save_steps = 100
    if training_args.save_total_limit is None:
        training_args.save_total_limit = 3


def save_run_config(output_dir, model_args, data_args, training_args, attn_implementation):
    path = Path(output_dir) / "run_config.json"
    config = {
        "model_name_or_path": model_args.model_name_or_path,
        "dataset_use": data_args.dataset_use,
        "seg_mask_size": data_args.seg_mask_size,
        "seg_enable": training_args.seg_enable,
        "seg_loss_weight": training_args.seg_loss_weight,
        "seg_box_expand": training_args.seg_box_expand,
        "seg_box_alpha": training_args.seg_box_alpha,
        "seg_use_highres_fusion": training_args.seg_use_highres_fusion,
        "seg_refine": training_args.seg_refine,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "num_train_epochs": training_args.num_train_epochs,
        "max_steps": training_args.max_steps,
        "logging_steps": training_args.logging_steps,
        "save_steps": training_args.save_steps,
        "save_total_limit": training_args.save_total_limit,
        "attn_implementation": attn_implementation,
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def train(attn_implementation="auto"):
    global local_rank
    attn_implementation = resolve_attn_implementation(attn_implementation)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    if training_args.seg_enable:
        if data_args.data_flatten or data_args.data_packing:
            raise ValueError("segmentation training does not support data_flatten or data_packing")
        if training_args.lora_enable:
            raise ValueError("phase 1 segmentation training does not support LoRA")
        configure_seg_defaults(training_args)
    save_run_config(training_args.output_dir, model_args, data_args, training_args, attn_implementation)

    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        if training_args.seg_enable:
            raise ValueError("phase 1 segmentation training only supports dense Qwen3-VL")
        from transformers import Qwen3VLMoeForConditionalGeneration

        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        from transformers import Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        from trainer import replace_qwen2_vl_attention_class

        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing and not training_args.seg_enable:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.seg_enable:
        if data_args.model_type != "qwen3vl":
            raise ValueError("segmentation training only supports qwen3vl")
        model = Qwen3VLSegForConditionalGeneration(
            model,
            seg_mask_size=data_args.seg_mask_size,
            seg_loss_weight=training_args.seg_loss_weight,
            seg_box_expand=training_args.seg_box_expand,
            seg_box_alpha=training_args.seg_box_alpha,
            seg_use_highres_fusion=training_args.seg_use_highres_fusion,
            seg_refine=training_args.seg_refine,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                rank0_print("Segmentation enabled: trainable mask head will be initialized on first forward.")
        else:
            rank0_print("Segmentation enabled: trainable mask head will be initialized on first forward.")
    elif training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer_cls = TongueSegTrainer if training_args.seg_enable else Trainer
    callbacks = [JsonlLogCallback(training_args.output_dir)] if training_args.seg_enable else None
    trainer = trainer_cls(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )

    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints and training_args.seg_enable:
        checkpoints = [path for path in checkpoints if (path / "trainer_state.json").exists()]
    if checkpoints:
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
