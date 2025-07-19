from diffusers import StableDiffusionPipeline
import torch

# Log in to Hugging Face Hub if necessary
# from huggingface_hub import login
# login("YOUR_HF_TOKEN")

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "A futuristic city skyline at sunset"
image = pipe(prompt).images[0]
image.save("generated_image.png")
