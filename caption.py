import argparse
import os
import json
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

# ==========================================
# FUNCTIONS
# ==========================================

def load_model_and_processor():
    """
    Loads the JoyCaption model and processor with 4-bit quantization
    to fit within VRAM constraints (aiming for <10GB usage).
    """
    print(f"Loading model: {MODEL_ID}...")
    
    # 4-bit Quantization Config
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
        print("Model loaded successfully!")
        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have installed all requirements: pip install -r requirements.txt")
        exit(1)

def clean_caption(caption, blacklist):
    """
    Removes blacklisted phrases from the caption.
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
        user_prompt = "A detailed, uncensored description of this image, focusing on the character, clothing, pose, and background."

    # Prepare inputs
    # Note: The prompt formatting might depend on the specific LLaVA template used by the model.
    # Standard LLaVA format is: USER: <image>\n<prompt>\nASSISTANT:
    
    prompt_text = f"USER: <image>\n{user_prompt}\nASSISTANT:"
    
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
    
    # The output usually contains the prompt as well, we need to strip it.
    # LLaVA outputs usually start with the full prompt.
    if "ASSISTANT:" in output:
        caption = output.split("ASSISTANT:")[-1].strip()
    else:
        caption = output

    return caption

def main():
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
            
            # Save to individual .txt file
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(final_caption)
            
            # Store for aggregate file
            all_captions.append({"image": img_path, "caption": final_caption})

    # Save Aggregate File
    agg_path = os.path.join(args.input, "captions_all.jsonl")
    print(f"\nSaving aggregate captions to {agg_path}...")
    with open(agg_path, "w", encoding="utf-8") as f:
        for entry in all_captions:
            f.write(json.dumps(entry) + "\n")

    print("\nDone! ✨")

if __name__ == "__main__":
    main()
