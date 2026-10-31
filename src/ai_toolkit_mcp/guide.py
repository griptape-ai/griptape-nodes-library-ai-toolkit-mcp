"""Best-practice notes served by the training_guide tool.

General LoRA advice plus ai-toolkit specifics (several taken from the comments in ai-toolkit's
config/examples). Edit freely as you learn what works.
"""

GUIDE: dict[str, str] = {
    "overview": """\
# Training a LoRA with ai-toolkit: the workflow

1. Check the machine: ai_toolkit_status (GPU VRAM, other jobs on the queue, whether an HF token is set).
2. Dataset: create_dataset from a folder of images (or clips). See topic "dataset".
3. Captions: get_captions to inspect, then set_captions or auto_caption_dataset. See topic "captions".
4. Model: list_models, then pick one that fits the goal and the GPU. See topics "models" and "vram".
5. Settings: create_job(model=..., overrides={...}). See topic "settings". Read the warnings it returns.
6. Confirm with the user, then start_job. Training takes minutes to hours.
7. Monitor: get_job / get_job_log while it starts up, then view_samples at each sample interval.
   See topic "monitoring".
8. Choose a checkpoint: list_job_outputs. The last one isn't always the best.

Rules of thumb:
- Ask before starting or stopping a job. It ties up the GPU, often for hours.
- Each GPU runs one job at a time; others wait in its queue.
- To train a model you haven't trained before, a job set up once in the ai-toolkit UI and cloned with
  create_job(base_job=...) is the most reliable path. The model presets get close but the UI knows
  every model's quirks.
- Topics: overview, dataset, captions, models, settings, vram, monitoring, troubleshooting.
""",
    "dataset": """\
# Building a dataset

How many images:
- A person, character, pet or object: 15-30 good images. 10 can work; more than 50 rarely helps.
- A style: 30-100+ images showing the style across many different subjects.
- Quality and variety beat quantity. Ten great images beat forty near-duplicates.

What makes a good set:
- Variety in everything you DON'T want the LoRA to learn: backgrounds, lighting, clothing, angles,
  framing (close-ups, medium shots, full body). Whatever stays constant gets learned as part of the concept.
- Consistency in what you DO want learned (the face, the product, the brushwork).
- Sharp, well-exposed, high resolution: ideally 1024px+ on the short side, at least 512px.
  ai-toolkit resizes and buckets images automatically, so mixed sizes and aspect ratios are fine;
  don't pre-crop to squares.
- Remove watermarks, text overlays, heavy filters, and other people/subjects you don't want learned
  (crop them out or drop the image).

Video models (LTX-2, Wan, etc.):
- Short clips (roughly 2-5 seconds) that show the motion or subject clearly. Set datasets.0.num_frames
  for how many frames to sample per clip; more frames = much more VRAM.
- Still images also work for teaching appearance on most video models (num_frames 1).

Files and folders:
- create_dataset uploads files or folders from this machine. Matching caption files come along
  (photo1.jpg + photo1.txt). ai-toolkit replaces spaces/special characters in file names with "_".
- Use the dataset's `path` from list_datasets/create_dataset as datasets.0.folder_path.
""",
    "captions": """\
# Captions and trigger words

How ai-toolkit uses them:
- Each image's caption is a .txt file with the same name (image2.jpg + image2.txt).
- trigger_word (in process[0], set via overrides {"trigger_word": "..."}) is added to any caption that
  doesn't already contain it. You can also write [trigger] in captions or sample prompts and ai-toolkit
  substitutes the trigger word.
- caption_dropout_rate (default 0.05) trains 5% of steps with no caption, which helps the LoRA work
  with short prompts.
- IMPORTANT: trigger words do not work when train.cache_text_embeddings is true (some models' presets
  turn it on to save VRAM). In that case, either don't use a trigger word or switch to
  train.unload_text_encoder instead (never both).

Trigger word:
- Use a short, made-up token unlikely to already mean something: "rufus_dog", "zk7style", "ohwx".
  Avoid real words or names the model already knows strongly.
- For a style LoRA a trigger word is optional; for a specific person/character/object, use one.

Writing captions (or reviewing auto-captions):
- Describe what VARIES and should stay controllable: setting, pose, clothing, lighting, camera angle,
  expression. The model attributes whatever is NOT described to the trigger word.
- For a person/character, don't describe their permanent identifying features in detail (face shape,
  eye color); let the trigger word absorb them. Do describe things you want to change later
  (outfit, hairstyle if it varies).
- For a style, describe the content of each image plainly and let the style be learned implicitly
  (or mention the trigger word as the style).
- Natural sentences suit modern models (FLUX, Qwen, Wan, LTX, Z-Image). Comma-separated tags suit
  SD1.5/SDXL anime-style models.
- auto_caption_dataset gives a solid first draft. Spot-check a few with get_captions and fix any that
  are wrong or that over-describe the subject.
""",
    "models": """\
# Choosing a model

- list_models shows every model this ai-toolkit install supports, grouped as image, video, instruction
  (edit models), audio, llm, experimental. Pass its `name` to create_job(model=...).
- Pick the model the user will generate with. A LoRA only works with the model family it was
  trained on.
- Common picks: FLUX.1 / Flex for general images; Qwen-Image for prompt adherence and text; Z-Image
  for fast, high quality images; SDXL for the large existing ecosystem; LTX-2.x and Wan for video.
- Gated models (list_models shows gate_url) need the user to accept the license on Hugging Face and
  an HF token saved in the ai-toolkit UI's Settings. Without it the job fails while downloading.
- Instruction/edit models (Kontext, Qwen-Image-Edit, etc.) train on before/after pairs with control
  images. Set those up in the ai-toolkit UI once and clone them; the presets don't configure controls.
- The first run of a model downloads it (often tens of GB), so the first job looks slow to start.
- Presets apply the same settings the UI does when you pick that model. Still read the resulting
  config (get_job_config) for anything that looks off.
""",
    "settings": """\
# Key settings (dotted paths for create_job/update_job overrides, inside config.process[0])

- train.steps: ai-toolkit suggests 500-4000. Around 2000-3000 suits 15-30 images. Fewer images or a
  simple concept: fewer steps. Styles and complex concepts: more. Since start_job resumes from the
  last checkpoint, you can raise steps later and continue instead of guessing high.
- train.lr: 1e-4 is the safe default. 2e-4 learns faster but overfits sooner. Lower it if samples
  "fry" (oversaturated, artifacts) early.
- network.linear (rank) and network.linear_alpha: 16-32 is typical. 16 is plenty for a person or
  object on large models; 32 for styles or complex concepts. Keep alpha equal to rank.
- train.content_or_style: "content" for subjects (people, objects), "style" for styles, "balanced"
  otherwise.
- train.batch_size: leave at 1 unless VRAM is plentiful; raise lr slightly if you increase it.
- save.save_every (default 250) and save.max_step_saves_to_keep (4): keep several checkpoints so
  you can pick an earlier one if the last overfits. Raise max_step_saves_to_keep for long runs.
- sample.sample_every (default 250) and sample.samples: samples take real time, so don't sample too
  often. Include the trigger word ([trigger]) in most prompts, and add one or two prompts that put
  the subject somewhere NEW (a setting/style not in the dataset) to detect overfitting. Add one
  prompt WITHOUT the trigger word to check the base model isn't being damaged.
- datasets.0.resolution: [512, 768, 1024] by default (multi-resolution helps most models). Drop 1024
  if you run out of VRAM.
- datasets.0.caption_dropout_rate: 0.05 default is fine.
""",
    "vram": """\
# Fitting in VRAM

Check free VRAM with ai_toolkit_status first; another job or app may be using the GPU.

Levers, roughly in order:
1. model.quantize / model.quantize_te true (most presets do this already).
2. model.low_vram true: slower but lighter, also recommended when the GPU drives your monitors.
3. model.layer_offloading true (models that support it): offloads layers to system RAM.
4. Lower datasets.0.resolution (drop 1024), and for video lower datasets.0.num_frames.
5. train.cache_text_embeddings true or train.unload_text_encoder true (not both) frees the text
   encoder, but caching breaks trigger words (see "captions").
6. Keep train.gradient_checkpointing true and train.batch_size 1.

Rough guide: most image models train on 24 GB with quantization. 14B+ video models at 24 GB need the
low-VRAM presets (low-bit quantization with accuracy recovery adapters). 48 GB+ is comfortable for
nearly everything.
""",
    "monitoring": """\
# Monitoring a run

- Right after start_job, check get_job_log: the first minutes are model download/load and latent
  caching. Errors usually appear here (missing token, OOM, bad path).
- get_job shows step progress and speed; estimate time left from those.
- Loss (get_loss) for these flow-matching models is noisy and doesn't track quality well. Use it to
  spot problems (NaN, sudden spikes, a jump after resuming), not to choose checkpoints.
- Judge by samples (view_samples), compared step to step:
  - Learning: the subject/style appears more clearly with the trigger word.
  - Well trained: strong likeness, but prompts still control setting, pose, clothing, style.
  - Overfitting: samples copy training backgrounds/poses, ignore the prompt, look oversaturated or
    noisy, or the no-trigger prompt starts showing the subject. Use an earlier checkpoint, or retrain
    with fewer steps or a lower lr.
  - Underfitting at the end: weak likeness. Raise train.steps with update_job (stop first) and
    start_job again to continue from the last checkpoint.
- trigger(job, "sample") or trigger(job, "save") gets samples or a checkpoint right now.
""",
    "troubleshooting": """\
# Common problems

- Job fails instantly with ModuleNotFoundError: ai-toolkit's Python environment isn't installed. Run
  its install (see ai-toolkit README) so a venv exists in the ai-toolkit folder.
- 401 / gated repo / "access to model is restricted": accept the model's license on Hugging Face and
  save an HF token in the ai-toolkit UI's Settings.
- CUDA out of memory: see topic "vram". Also check whether another job or app holds the GPU.
- Job stuck "queued": another job is running on that GPU's queue (one at a time), or the queue was
  stopped. start_job restarts the queue.
- Samples look like the base model even late in training: trigger word missing from sample prompts,
  trigger words disabled by cache_text_embeddings, or the dataset path points to the wrong folder.
- Stop doesn't finish: stop_job gives the trainer time to save. Use force=True only if it's hung.
- Trigger word ignored at generation time: captions may describe the subject so fully that the
  trigger word learned nothing. Simplify the captions (see "captions").
""",
}

TOPICS = tuple(GUIDE)
