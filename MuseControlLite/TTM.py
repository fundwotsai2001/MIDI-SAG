import torch
import soundfile as sf
from diffusers import StableAudioPipeline

pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# define the prompts
BACKING_TEXT_PROMPTS=[
  "reflective instrumental pop with piano, synth pad, bass, and steady drums",
  "groovy funk with bass guitar, electric guitar, drums, and electric piano",
  "intense instrumental rock with electric guitar riffs, bass guitar, drums, and powerful energy",
  "melancholic electronic backing with piano, bass, reflective synth textures, and instrumental pop style"
]

negative_prompt = ["Low quality.","Low quality.","Low quality.","Low quality."]

# set the seed for generator
generator = torch.Generator("cuda").manual_seed(0)

# run the generation
audio = pipe(
    BACKING_TEXT_PROMPTS,
    negative_prompt=negative_prompt,
    num_inference_steps=200,
    audio_end_in_s=10.0,
    num_waveforms_per_prompt=3,
    generator=generator,
).audios

output = audio[0].T.float().cpu().numpy()
sf.write("hammer.wav", output, pipe.vae.sampling_rate)
