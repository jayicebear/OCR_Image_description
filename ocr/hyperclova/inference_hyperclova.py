from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image
import time
import torch
import os

def describe_image_with_hyperclovax(
    image_path: str,
    model_name: str = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
    max_new_tokens: int = 3000,
    temperature: float = 0.5,
    top_p: float = 0.6,
    device: str = "cuda",
):
    """
    Generate a concise image description using HyperCLOVAX-SEED-Vision-Instruct-3B.

    Args:
        image_path (str): Path to the image file.
        model_name (str): HuggingFace model name or local path.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature for generation.
        top_p (float): Nucleus sampling parameter.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        dict: {
            'output': generated_description,
            'elapsed_time': execution_time_seconds
        }
    """

    # --- Start timer ---
    start_time = time.time()

    # --- Load model and processor ---
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Load image ---
    image = Image.open(image_path).convert("RGB")

    # --- Prompts ---
    system_prompt = "You are a helpful image analyzer"
    user_prompt = """
Find images in the file and provide the following:
**Image Description**: Briefly describe the visual content (objects, scenes, etc.).
If there is no image (only text), output: NO

**Important notes**:
**Keep it concise and do not repeat**
"""

    # --- Chat structure ---
    vlm_chat = [
        {"role": "system", "content": [{"text": system_prompt, "type": "text"}]},
        {"role": "user", "content": [{"text": user_prompt, "type": "text"}]},
        {
            "role": "user",
            "content": [
                {
                    "filename": image_path.split("/")[-1],
                    "image": image,
                    "type": "image",
                }
            ],
        },
    ]

    # --- Prepare model input ---
    model_inputs = processor.apply_chat_template(
        vlm_chat,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)

    # --- Generate output ---
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=1.5,
    )

    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    output_text = output_text.replace("<|im_end|><|endofturn|>", "")

    elapsed = time.time() - start_time

    return {"output": output_text.strip(), "elapsed_time": elapsed}


# Example usage
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    file_name = "테스트"
    image_path = f"/home/ljm/ocr/picture/{file_name}.png"
    result = describe_image_with_hyperclovax(image_path)
    print("=" * 80)
    print("VLM Example Result:")
    print(result["output"])
    print("=" * 80)
    print("Elapsed time:", result["elapsed_time"], "seconds")
