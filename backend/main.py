import os
import modal
import uuid
import base64
from pydantic import BaseModel
import requests

app = modal.App("music-generator")


image = (
    modal.Image.debian_slim().
    apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step", " cd /tmp/ACE-Step && pip install ."])
    .env({"HF_HOME": "/.cache/huggingface"})
    .add_local_python_source("prompts")
)

model_volume = modal.Volume.from_name(
    "ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

music_gen_secrets = modal.Secret.from_name("music-gen-secret")


class GenerateMusicResponse(BaseModel):
    audio_data: str
    #   Add other fields as necessary


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/modles": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=10,  # stay longer after last use in seconds
)
class MusicGenServer:
    @modal.enter()
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from diffusers import AutoPipelineForText2Image
        import torch
        self.music_model = ACEStepPipeline(
            checkpoint_dir="/models",
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )

        # define LLM model
        model_id = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm_model = model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface",
        )

        # stable diffusion model for thumbnail generation
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="/.cache/huggingface"
        )
        pipe.to("cuda")

    @modal.fastapi_endpoint(method="POST")
    def generate(self) -> GenerateMusicResponse:
        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4}.wav")

        self.music_model(
            prompt="A calming piano melody with soft strings in the background",
            lyrics="A soothing tune to relax your mind as you drift into sleep  under the starry sky.  Let the gentle notes wash over you, bringing peace and tranquility to your soul. Feel the stress melt away with each chord, as the music guides you to a place of serenity and calm. Close your eyes and let the melody carry you to a world of dreams, where worries fade and only harmony remains. Embrace the night with this peaceful lullaby, a perfect soundtrack for restful sleep and sweet dreams.  ",
            audio_duration=180,
            infer_step=60,
            guidance_scale=15,
            save_path=output_path
        )
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        os.remove(output_path)

        return GenerateMusicResponse(audio_data=audio_base64)

    def test_endpoint(self):
        self.music_model


@app.local_entrypoint()
def main():
    print("This is the main function.")
    server = MusicGenServer()
    endpoint_url = server.generate.get_web_url()
    response = requests.post(endpoint_url)
    response.raise_for_status()
    result = GenerateMusicResponse(**response.json())
    audio_bytes = base64.b64decode(result.audio_data)
    output_filename = "generated_music.wav"
    with open(output_filename, "wb") as f:
        f.write(audio_bytes)
