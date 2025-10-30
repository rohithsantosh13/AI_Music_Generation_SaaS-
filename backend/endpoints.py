from prompts import PROMPT_GENERATOR_PROMPT, LYRICS_GENERATOR_PROMPT, THUMBNAIL_GENERATOR_PROMPT, CATOGERIES_GENERATOR_PROMPT
import os
import boto3
import modal
import uuid
from app import app
from typing import List
from audio_generation_models import GenerateMusicResponseS3, GenerateFromDescriptionRequest, GenerateWithCustomLyricsRequest, GenerateWithDescribedLyricsRequest

image = (
    modal.Image.debian_slim().
    apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step", " cd /tmp/ACE-Step && pip install ."])
    .env({"HF_HOME": "/.cache/huggingface"})
    .add_local_python_source("prompts")
    .add_local_python_source("audio_generation_models")
    .add_local_python_source("endpoints")
    .add_local_python_source("app")
)
model_volume = modal.Volume.from_name(
    "ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)
music_gen_secrets = modal.Secret.from_name("music-gen-secret")


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
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface",
        )

        # stable diffusion model for thumbnail generation
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="/.cache/huggingface"
        )
        self.image_pipe.to("cuda")

    def prompt_qwen(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer(
            [text], return_tensors="pt").to(self.llm_model.device)

        generated_ids = self.llm_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_prompts(self, description: str) -> str:
        full_prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt=description)
        return self.prompt_qwen(full_prompt)

    def generate_lyrics(self, description: str) -> str:
        full_prompt = LYRICS_GENERATOR_PROMPT.format(description=description)
        return self.prompt_qwen(full_prompt)

    def generate_categories(self, description: str) -> List[str]:
        categories_prompt = CATOGERIES_GENERATOR_PROMPT.format(
            description=description)
        response_text = self.prompt_qwen(categories_prompt)
        categories = [cat.strip()
                      for cat in response_text.split(",") if cat.strip()]
        return categories

    def generate_and_upload_to_S3(self, prompt: str,
                                  lyrics: str,
                                  instrumental: bool,
                                  audio_duration: int,
                                  infer_step: int,
                                  guidance_scale: int,
                                  seed: int,
                                  description: str
                                  ) -> GenerateMusicResponseS3:
        final_lyics = lyrics if not instrumental else "[instumental]"
        print("Generating music with prompt:",
              prompt, " and lyrics:", final_lyics)

        s3_client = boto3.client('s3')
        bucket_name = os.environ.get("S3_BUCKET_NAME")

        # Generate music
        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4}.wav")

        self.music_model(
            prompt=prompt,
            lyrics=final_lyics,
            audio_duration=audio_duration,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            save_path=output_path,
            manual_seeds=str(seed)
        )

        audio_s3_key = f"{uuid.uuid4}.wav"
        s3_client.upload_file(output_path, bucket_name, audio_s3_key)
        os.remove(output_path)

        # Generate thumbnail
        thumbnail_prompt = THUMBNAIL_GENERATOR_PROMPT.format(
            user_prompt=prompt)

        image = self.image_pipe(
            thumbnail_prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
        image_output_path = os.path.join(output_dir, f"{uuid.uuid4}.png")
        image.save(image_output_path)

        image_s3_key = f"{uuid.uuid4()}.png"
        s3_client.upload_file(image_output_path, bucket_name, image_s3_key)
        os.remove(image_output_path)

        # Generate categories
        categories = self.generate_categories(description=description)
        return GenerateMusicResponseS3(
            s3_key=audio_s3_key,
            cover_image_s3_key=image_s3_key,
            categories=categories
        )

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_from_description(self, request: GenerateFromDescriptionRequest) -> GenerateMusicResponseS3:
        # Generate a prompt using llm  and generate lyrics using the prompt
        prompt = self.generate_prompts(request.full_described_song)

        lyrisc = ""

        if not request.instrumental:
            lyrics = self.generate_lyrics(request.full_described_song)

        return self.generate_and_upload_to_S3(
            prompt=prompt,
            lyrics=lyrics,
            description=request.full_described_song,
            **request.model_dump(exclude={"full_described_song"})
        )

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_with_lyrics(self, request: GenerateWithCustomLyricsRequest) -> GenerateMusicResponseS3:
        return self.generate_and_upload_to_S3(
            prompt=request.prompt,
            lyrics=request.lyrics,
            description=request.prompt,
            **request.model_dump(exclude={"prompt", "lyrics"})
        )

    # Generate music using the generated prompt and lyrics
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_from_described_lyrics(self, request: GenerateWithDescribedLyricsRequest) -> GenerateMusicResponseS3:
        lyrics = ""

        if not request.instrumental:
            lyrics = self.generate_lyrics(request.described_lyrics)
        
        return self.generate_and_upload_to_S3(
            prompt=request.prompt,
            lyrics=lyrics,
            description=request.prompt,
            **request.model_dump(exclude={"described_lyrics", "prompt"})
        )
