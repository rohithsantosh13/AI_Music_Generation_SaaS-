from pydantic import BaseModel
from typing import List


class GenerateMusicResponse(BaseModel):
    audio_data: str
    #   Add other fields as necessary


class GenerateMusicResponseS3(BaseModel):
    s3_key: str
    cover_image_s3_key: str
    categories: List[str]


class AudioGenerationBase(BaseModel):
    audio_duration: float = 15.0  # in seconds
    seed: int = 42  # random seed for generation
    guidance_scale: float = 15.0  # guidance scale for diffusion
    infer_step: int = 60  # number of diffusion steps
    instrumental: bool = False  # whether to generate instrumental music


class GenerateFromDescriptionRequest(AudioGenerationBase):
    full_described_song: str


class GenerateWithCustomLyricsRequest(AudioGenerationBase):
    prompt: str
    lyrics: str


class GenerateWithDescribedLyricsRequest(AudioGenerationBase):
    prompt: str
    described_lyrics: str
