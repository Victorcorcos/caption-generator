# JoyCaption Generator

A powerful, local AI tool for generating rich, descriptive captions for anime and artwork images. This tool leverages the **JoyCaption** model (LLaVA-based) to provide detailed, uncensored descriptions suitable for training Stable Diffusion LoRAs or organizing large datasets.

## Features

- **Rich Descriptions:** Generates detailed sentences covering character appearance, pose, clothing, and background.
- **Uncensored:** Capable of handling mixed SFW and NSFW datasets without arbitrary filtering.
- **Batch Processing:** Recursively scans folders and subfolders.
- **Blacklist System:** Automatically removes unwanted keywords (e.g., specific hair colors) from generated captions.
- **Formats:** Outputs individual `.txt` files next to each image and a master `.jsonl` file.
- **Efficient:** Uses 4-bit quantization (bitsandbytes) to run on consumer GPUs (requires ~8-10GB VRAM).

## Requirements

- **OS:** Linux (Recommended) or Windows
- **GPU:** NVIDIA GPU with at least 10GB VRAM (12GB+ recommended for comfort).
- **Python:** 3.10+
- **CUDA:** 11.8 or 12.x

## Installation

1.  **Clone or Open the Repository:**
    ```bash
    cd /path/to/caption-generator
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script pointing to your image directory:

```bash
python caption.py -i /path/to/your/images
```

### Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `-i`, `--input` | Path to the folder containing images. | (Required) |
| `-r`, `--recursive` | Recursively search subfolders. | `True` |
| `--force` | Overwrite existing `.txt` caption files. | `False` |
| `--mode` | Caption style: `descriptive` (sentences) or `tags` (Booru list). | `descriptive` |

### Example

Caption a folder of images, forcing overwrite of old captions, and using descriptive mode:

```bash
python caption.py -i /home/user/pictures/anime --force --mode descriptive
```

## Configuration

### Blacklist
To prevent specific terms from appearing in your captions (e.g., if you want to hard-code a trigger word in your training and don't want the captioner to describe it redundantly, or if the model hallucinates a specific trait), edit the `BLACKLIST` array in `caption.py`:

```python
BLACKLIST = [
    "black hair",
    "brown hair",
    "watermark",
    "username"
]
```

## Model Details

This tool defaults to using `fancyfeast/llama-joycaption-alpha-two-hf-llava`. This is a HuggingFace-compatible implementation of the JoyCaption model series. It is loaded in 4-bit precision (NF4) to maximize performance on consumer hardware.

## License

This project is for personal and educational use. Please respect the licenses of the underlying models (JoyCaption, Llama 3, LLaVA).
