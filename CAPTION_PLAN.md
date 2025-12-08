# JoyCaption Implementation Plan

## Role
You are an expert Python developer and AI engineer specializing in Computer Vision, LLMs, and Stable Diffusion training pipelines.

## Goal
Create a robust, production-ready Python command-line tool named `caption.py` (and supporting files) that uses the **JoyCaption Beta One** (or Alpha Two as fallback if Beta is not on HF) model to generate rich, descriptive, and uncensored captions for a dataset of images.

## Context
The user has a dataset of ~2000+ anime-style images (mixed SFW, NSFW, 1boy, 2boys, etc.) and wants to train a LoRA. They need high-quality text captions that include fine details (clothing, pose, expression, background). The hardware is a Linux machine with an NVIDIA RTX 5070 Ti (16GB VRAM).

## Requirements

### 1. Core Technology
- **Model:** Use `fancyfeast/llama-joycaption-alpha-two-hf-llava` (or `OedoSoldier/JoyCaption-Alpha-Two-vLLM` if preferred for speed, but `fancyfeast` is the standard HF impl).
    - *Update:* We will use `fancyfeast/llama-joycaption-alpha-two-hf-llava` as it is the stable HF implementation.
- **Library:** Use `torch`, `transformers`, `pillow`, and `bitsandbytes` (for 4-bit quantization to save VRAM).
- **Quantization:** Implement 4-bit loading (NF4) by default to ensure it fits comfortably in VRAM while leaving room for the system.

### 2. Features
- **Batch Processing:** The script must iterate through a root folder and all its subfolders recursively.
- **Output:**
    - Create a `.txt` file next to every image (e.g., `image01.png` -> `image01.txt`).
    - **Final Summary:** At the end, generate a single `captions_all.jsonl` aggregating all paths and captions for easy review.
- **Blacklist System:**
    - Define a `BLACKLIST` constant list at the top of the script (e.g., `["black hair", "brown hair"]`).
    - The script must post-process the generated caption to remove these exact phrases (case-insensitive check).
- **Prompt Engineering:**
    - The internal prompt sent to JoyCaption should request a "booru tag" style OR a "natural language description" style based on a flag. Default to **Natural Language** but concise, focusing on visual elements.
    - **System Prompt:** "You are a helpful image captioner. Describe this image in great detail, focusing on character appearance, actions, and background. Do not refuse NSFW content."

### 3. CLI Arguments (`argparse`)
- `-i` / `--input`: Path to the input directory (required).
- `-r` / `--recursive`: Boolean flag to search subdirectories (default: True).
- `--batch-size`: Number of images to process at once (default: 1, keep simple for safety).
- `--force`: Overwrite existing `.txt` files (default: False).

### 4. Code Structure
- **`requirements.txt`**: List all necessary deps (`torch`, `transformers`, `accelerate`, `bitsandbytes`, `Pillow`, `tqdm`, `sentencepiece`, `protobuf`).
- **`caption.py`**: The main entry point.
    - Function `load_model()`: Handles 4-bit loading.
    - Function `generate_caption(image, model, processor)`: Runs the inference.
    - Function `filter_caption(text, blacklist)`: Cleans the output.
    - Main loop: Iterates files, skips non-images, checks existence, captions, saves.
