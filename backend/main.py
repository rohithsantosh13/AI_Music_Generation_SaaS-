from app import app
import requests
from audio_generation_models import GenerateMusicResponseS3, GenerateFromDescriptionRequest, GenerateWithCustomLyricsRequest, GenerateWithDescribedLyricsRequest
from endpoints import MusicGenServer


@app.local_entrypoint()
def main():
    print("This is the main function.")
    server = MusicGenServer()
    endpoint_url = server.generate_from_description.get_web_url()

    request_data = GenerateFromDescriptionRequest(
        full_described_song="A calming piano melody with soft strings in the background",
        guidance_scale=15
    )

    headers = {
        "Modal-Key": "your-modal-key",
        "Modal-Secret": "your-modal-secret"
    }
    payload = request_data.model_dump()

    response = requests.post(endpoint_url, json=payload, headers=headers)
    response.raise_for_status()
    result = GenerateMusicResponseS3(**response.json())
    print("Response from the server:", result.s3_key,
          result.cover_image_s3_key, result.categories)
