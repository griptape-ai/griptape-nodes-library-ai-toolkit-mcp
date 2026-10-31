import copy

# Ported from ai-toolkit ui/src/app/jobs/new/jobConfig.ts (defaultJobConfig).
_DEFAULT_JOB_CONFIG = {
    "job": "extension",
    "config": {
        "name": "my_first_lora_v1",
        "process": [
            {
                "type": "diffusion_trainer",
                "training_folder": "output",
                "device": "cuda",
                "trigger_word": None,
                "performance_log_every": 10,
                "network": {"type": "lora", "linear": 32, "linear_alpha": 32, "conv": 16, "conv_alpha": 16},
                "save": {
                    "dtype": "bf16",
                    "save_every": 250,
                    "max_step_saves_to_keep": 4,
                    "save_format": "diffusers",
                    "push_to_hub": False,
                },
                "datasets": [
                    {
                        "folder_path": "/path/to/images/folder",
                        "caption_ext": "txt",
                        "default_caption": "",
                        "caption_dropout_rate": 0.05,
                        "cache_latents_to_disk": False,
                        "resolution": [512, 768, 1024],
                        "num_frames": 1,
                        "num_repeats": 1,
                    }
                ],
                "train": {
                    "batch_size": 1,
                    "steps": 3000,
                    "gradient_accumulation": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "optimizer": "adamw8bit",
                    "timestep_type": "sigmoid",
                    "content_or_style": "balanced",
                    "optimizer_params": {"weight_decay": 1e-4},
                    "lr": 0.0001,
                    "dtype": "bf16",
                },
                "logging": {"log_every": 1, "use_ui_logger": True},
                "model": {
                    "name_or_path": "ostris/Flex.1-alpha",
                    "arch": "flex1",
                    "quantize": True,
                    "qtype": "qfloat8",
                    "quantize_te": True,
                    "qtype_te": "qfloat8",
                    "low_vram": False,
                },
                "sample": {
                    "sampler": "flowmatch",
                    "sample_every": 250,
                    "width": 1024,
                    "height": 1024,
                    "samples": [
                        {"prompt": "a woman holding a coffee cup, in a beanie, sitting at a cafe"},
                        {"prompt": "a bear building a log cabin in the snow covered mountains"},
                    ],
                    "neg": "",
                    "seed": 42,
                    "walk_seed": True,
                    "guidance_scale": 4,
                    "sample_steps": 30,
                    "num_frames": 1,
                    "fps": 1,
                },
            }
        ],
    },
    "meta": {"name": "[name]", "version": "1.0"},
}


def default_job_config() -> dict:
    return copy.deepcopy(_DEFAULT_JOB_CONFIG)
