import torch
from diffusers import StableDiffusionPipeline

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained Stable Diffusion model outside the function
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe.to(device)  # Use GPU if available

def generate_image_from_prompt(prompt):
    """
    Generates an image from a given prompt using Stable Diffusion.

    Args:
        prompt: The text prompt to guide image generation.

    Returns:
        None. Saves the generated image to "generated_image.png".
    """
    try:
        # Generate the image
        with torch.no_grad():
            image = pipe(prompt).images[0]

        # Save or display the image
        image.save("generated_image.png")
        print("Image saved successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

# Get user input for the prompt
user_prompt = input("Enter your image prompt: ")

# Call the function to generate the image
generate_image_from_prompt(user_prompt)
