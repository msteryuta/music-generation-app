from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torchaudio

# Load processor and model
processor = AutoProcessor.from_pretrained("facebook/musicgen-stereo-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-stereo-small").to("cuda")

# Prepare inputs
inputs = processor(
    text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    padding=True,
    return_tensors="pt",
)

# Move inputs to the GPU
inputs = {key: value.to("cuda") for key, value in inputs.items()}

# Generate audio
audio_values = model.generate(**inputs, max_length=1000)  # Use max_length instead of max_new_tokens

# Save or process the audio
for idx, audio in enumerate(audio_values):
    torchaudio.save(f"generated_audio_{idx}.wav", audio.cpu(), sample_rate=16000)

print("Audio files generated successfully!")
