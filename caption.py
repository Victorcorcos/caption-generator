import argparse
import os
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

# Words/Phrases to exclude from the generated captions (Case Insensitive)
BLACKLIST = [
    "black hair",
    "brown hair",
    # Add more blacklist terms here
]

# Model ID on HuggingFace
# Using the Alpha Two HF LLaVA compatible version which is stable and high quality.
# You can swap this for a Beta version if a compatible HF repo becomes available.
MODEL_ID = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
AGGREGATE_FILENAME = "captions_all.txt"

# ==========================================
# FUNCTIONS
# ==========================================

def disable_siglip_pooling_head(model):
    """
    Disables the SigLIP vision pooling head that relies on `nn.MultiheadAttention`,
    because bitsandbytes cannot quantize that module which leads to dtype mismatches.

    Args:
        model (torch.nn.Module): The loaded Llava model instance.

    Returns:
        None
    """
    vision_tower = getattr(model, "vision_tower", None)
    if vision_tower is None:
        return

    vision_model = getattr(vision_tower, "vision_model", None)
    use_head = getattr(vision_model, "use_head", False) if vision_model else False
    if not use_head:
        return

    vision_model.use_head = False
    vision_model.head = None
    print("Disabled SigLIP pooling head to keep 4-bit quantization stable.")


def load_model_and_processor():
    """
    Loads the JoyCaption model and processor with 4-bit quantization to
    fit within VRAM constraints (aiming for <10GB usage).

    Returns:
        tuple[LlavaForConditionalGeneration, AutoProcessor]: The loaded model
        ready on the appropriate device along with its paired processor.
    """
    print(f"Loading model: {MODEL_ID}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        disable_siglip_pooling_head(model)
        print("Model loaded successfully!")
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have installed all requirements: pip install -r requirements.txt")
        exit(1)


def build_chat_prompt(processor, user_prompt):
    """
    Builds a chat-formatted prompt ensuring that image placeholder tokens are present.

    Args:
        processor (AutoProcessor): The processor tied to the JoyCaption model.
        user_prompt (str): Instructions for the assistant regarding the image.

    Returns:
        str: A prompt string ready for the tokenizer that includes image slots.
    """
    image_token = getattr(processor, "image_token", "<image>")
    # The FancyFeast template expects plain text that already embeds the image
    # placeholder token, so we combine the image tag and instructions up front.
    conversation = [
        {
            "role": "user",
            "content": f"{image_token}\n{user_prompt}",
        }
    ]

    apply_template = getattr(processor, "apply_chat_template", None)
    if callable(apply_template):
        try:
            return processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        except Exception as error:
            print(f"Warning: failed to apply processor chat template ({error}). Falling back to manual prompt.")

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(tokenizer_template):
            try:
                return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            except Exception as error:
                print(f"Warning: failed to apply tokenizer chat template ({error}). Falling back to manual prompt.")

    return f"USER: {image_token}\n{user_prompt}\nASSISTANT:"


def extract_assistant_response(raw_output):
    """
    Returns only the assistant portion of a generated conversation transcript.

    Args:
        raw_output (str): The decoded conversation that may include system/user turns.

    Returns:
        str: The assistant response, or the original text when no marker is found.
    """
    if not raw_output:
        return ""

    lines = raw_output.splitlines()
    response_lines = []
    collecting = False

    for line in lines:
        if line.strip().lower() == "assistant":
            collecting = True
            response_lines = []
            continue
        if collecting:
            response_lines.append(line)

    if response_lines:
        return "\n".join(response_lines).strip()

    marker_pattern = re.compile(r"\bassistant\b[:\s]*", re.IGNORECASE)
    last_match = None
    for match in marker_pattern.finditer(raw_output):
        last_match = match

    if last_match:
        return raw_output[last_match.end():].strip()

    return raw_output.strip()

def clean_caption(caption, blacklist):
    """
    Removes the provided blacklist phrases from a generated caption.

    Args:
        caption (str): The raw caption returned by the model.
        blacklist (list[str]): Terms that must be pruned from the caption.

    Returns:
        str: The sanitized caption with forbidden phrases removed.
    """
    cleaned = caption
    for phrase in blacklist:
        # Case insensitive replace
        pattern = phrase.lower()
        if pattern in cleaned.lower():
            # A simple replace might be tricky with casing, so we do a case-insensitive check
            # and remove it. For robustness, we can use regex, but simple replacement 
            # of the exact match in the original string is better if we find the index.
            # Here we will do a simple pass.
            
            # Find start index
            lower_cleaned = cleaned.lower()
            start_idx = lower_cleaned.find(pattern)
            while start_idx != -1:
                end_idx = start_idx + len(pattern)
                # Remove the phrase and any immediate following comma/space if it creates a double separator
                cleaned = cleaned[:start_idx] + cleaned[end_idx:]
                
                # Re-check for next occurrence
                lower_cleaned = cleaned.lower()
                start_idx = lower_cleaned.find(pattern)
    
    # Post-cleanup to fix double spaces or commas
    cleaned = cleaned.replace(" ,", ",").replace(",,", ",").strip()
    return cleaned

def generate_caption(image_path, model, processor, prompt_type="descriptive"):
    """
    Generates a caption for a single image.

    Args:
        image_path (str): Absolute or relative path to the image file.
        model (LlavaForConditionalGeneration): The LLaVA model instance.
        processor (AutoProcessor): Preprocessor that formats text and images.
        prompt_type (str): Either 'descriptive' or 'tags' to steer the prompt.

    Returns:
        str | None: The generated caption, or None when the image cannot be read.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open image {image_path}: {e}")
        return None

    # JoyCaption User Prompt
    # We can adjust this prompt to steer the model towards tags or sentences.
    if prompt_type == "tags":
        user_prompt = "Write a list of Booru-style tags for this image."
    else:
        user_prompt = "A detailed description of this image, focusing on the character, clothing, pose, and background."

    prompt_text = build_chat_prompt(processor, user_prompt)

    inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1
        )

    # Decode
    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return extract_assistant_response(output)

def main():
    """
    Parses CLI arguments and orchestrates caption generation for folders.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="JoyCaption Batch Processor")
    parser.add_argument("-i", "--input", required=True, help="Path to input directory containing images")
    parser.add_argument("-r", "--recursive", action="store_true", default=True, help="Scan subdirectories recursively (default: True)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .txt files")
    parser.add_argument("--mode", choices=["descriptive", "tags"], default="descriptive", help="Caption style: 'descriptive' (sentences) or 'tags' (Booru tags)")
    
    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist.")
        return

    # Gather images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = []

    print(f"Scanning '{args.input}' for images...")
    for root, dirs, files in os.walk(args.input):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
        
        if not args.recursive:
            break

    print(f"Found {len(image_files)} images.")
    if len(image_files) == 0:
        return

    # Load Model
    model, processor = load_model_and_processor()

    # Processing Loop
    all_captions = []
    
    print("\nStarting caption generation...")
    for img_path in tqdm(image_files, desc="Captioning"):
        txt_path = os.path.splitext(img_path)[0] + ".txt"

        # Check if exists
        if os.path.exists(txt_path) and not args.force:
            continue

        # Generate
        raw_caption = generate_caption(img_path, model, processor, args.mode)
        
        if raw_caption:
            # Clean
            final_caption = clean_caption(raw_caption, BLACKLIST)
            final_caption = " ".join(final_caption.split())
            
            # Save to individual .txt file
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"{final_caption}\n")
            
            # Store for aggregate file
            all_captions.append(final_caption)

    # Save Aggregate File
    agg_path = os.path.join(args.input, AGGREGATE_FILENAME)
    print(f"\nSaving aggregate captions to {agg_path}...")
    with open(agg_path, "w", encoding="utf-8") as f:
        for caption in all_captions:
            f.write(f"{caption}\n")

    print("\nDone! ✨")

if __name__ == "__main__":
    main()
