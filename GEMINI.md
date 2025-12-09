# Gemini AI Guidelines for Caption Generator

## Project Overview
This project is a local AI tool for generating rich, descriptive captions or Booru-style tags for anime and artwork images using the JoyCaption model (LLaVA-based). It supports batch processing, custom blacklists, and efficient 4-bit quantization.

## Tech Stack
- **Language:** Python 3.10+
- **ML Framework:** PyTorch
- **Libraries:**
    - `transformers` (Hugging Face)
    - `bitsandbytes` (Quantization)
    - `Pillow` (Image processing)
    - `tqdm` (Progress bars)

## Coding Standards

### Style
- Follow **PEP 8** guidelines.
- Use 4 spaces for indentation.
- **Naming:**
    - Variables/Functions: `snake_case`
    - Classes: `PascalCase`
    - Constants: `UPPER_SNAKE_CASE`
- **Imports:** Group standard library first, then third-party, then local imports.

### Documentation
- All functions must have a docstring explaining arguments, return values, and purpose.
- Comments should explain *why*, not just *what*.

### Structure
- Keep configuration constants (like `BLACKLIST`, `MODEL_ID`) at the top of `caption.py`.
- Separate logic into focused functions (`load_model_and_processor`, `clean_caption`, `generate_caption`).
- Use `if __name__ == "__main__":` for the entry point.

## Clean Code Techniques

Always apply these core principles when working with this codebase:

1.  **DRY (Don't Repeat Yourself)**: If code is identical or very similar, extract it into a generalized function. Parameters are your friends.
2.  **KISS (Keep It Simple Stupid)**: Make code so "stupid" that a 5-year-old could understand it.
3.  **SRP (Single Responsibility Principle)**: Separate code into simple, well-defined, well-intentioned tasks with clear names. Prevents "spaghetti code".
4.  **Avoid Hadouken IFs**: Avoid nested IFs → Solution: Early Returns and/or Switch-Cases.
5.  **Avoid Negative Conditionals**: Positive conditionals reduce mental strain and make code easier to reason about.
6.  **Encapsulate Conditionals**: For conditions with 3+ comparisons, extract into functions that convey the intent. Create names that reveal the conditional's purpose.
7.  **Avoid Flag Arguments**: Avoid boolean arguments (true/false) to functions. Use descriptive strings or enums instead.
8.  **Avoid Comments**: Code should be self-documenting with intention-revealing names. If comments are necessary, explain the "why" not the "what". Use SRP and intention-revealing names as your primary tools.
9.  **Good Nomenclatures**: Use descriptive variable names that reveal intent. Use pronounceable and searchable names. Follow language, business, and team conventions.
10. **Use Vertical Formatting**: Code should read top to bottom without "jumping". Similar and dependent functions should be vertically close.
11. **Boy Scout Rule**: Always leave the codebase cleaner than you found it. Improve Clean Code whenever you touch existing code.

## Workflow
1.  **Environment:** Always work within the `venv`.
2.  **Dependencies:** Update `requirements.txt` if new packages are added.
3.  **Testing:** Since this is a script, test manually by running on a small sample folder of images.
    - Example: `python caption.py -i ./test_images --force`

## Key Implementation Details
- **Model:** `fancyfeast/llama-joycaption-alpha-two-hf-llava`
- **Quantization:** NF4 (4-bit) via `BitsAndBytesConfig`.
- **Outputs:**
    - `.txt` file next to each image.
    - `captions_all.jsonl` summary file in the root of the input directory.

## Future Improvements
- Add type hints (`from typing import ...`).
- Implement logging instead of `print` statements.
- Add unit tests for `clean_caption`.
