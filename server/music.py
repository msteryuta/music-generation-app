import requests

API_URL = "https://api-inference.huggingface.co/models/facebook/musicgen-small"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

audio_bytes = query({
	"inputs": "liquid drum and bass, atmospheric synths, airy sounds",
})
# You can access the audio with IPython.display for example
from IPython.display import Audio
Audio(audio_bytes)