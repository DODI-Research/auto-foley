<h1 style='text-align: center; margin-bottom: 1rem'> Auto-Foley </h1>

<div style="display: flex; flex-direction: row; justify-content: center">
<a href="https://dodi-research.github.io/projects" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/project_page-white?logo=github&logoColor=black"></a>
<a href="https://github.com/DODI-Research/auto-foley" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/github-white?logo=github&logoColor=black"></a>
<a href="https://huggingface.co/spaces/DODI-Research/auto-foley-editor" target="_blank"><img alt="Open in Spaces" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg"></a>
</div>

Auto-Foley is a proof-of-concept for the use of large language models in environments where they form a soft agentic approach instead of in the typical text-in & text-out usage.

With this example, we present how to combine the vision capabilities of a large language model with a sound effect generator to get a working video-to-audio workflow.

Auto-Foley extracts samples from an input video and then provides them to a vision large language model with the instructions to identify potential audio sources in the video scene. The vision large language model then returns a structured list of prompts which are sent to a text-to-sound-effects service.
The resulting generated audio files are then combined with the input video

## Quick Start
Example for Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

This repository is no AI model but rather a set of instructions for two different AI model services.
Auto-Foley comes with one default implmenetation for each service. 
OpenAI's ChatGPT API for vision large languge model.
And ElevenLabs text to sound effects API for text to sound effects.

You are required to supply your own API keys if you wish to use these default implementations.
You can either save them in your environment variables:
- `AUTO_FOLEY_DEFAULT_VISION_LM_API_KEY`: Default API key for vision language model
- `AUTO_FOLEY_DEFAULT_TTSFX_API_KEY`: Default API key for text-to-speech effects

In which case you can now run Auto-Foley like this:
```bash
python run_auto_foley.py -i input.mp4
```
Alternatively, you can pass the API keys directly, along with other optional parameters:
```bash
python run_auto_foley.py 
    --input input.mp4
    --output output.mp4
    --frame-interval 25
    --downscale-max-side 512
    --prompt-instruction "Focus on mechanical sounds"
    --vision-lm-api-key YOUR_OPENAI_KEY
    --ttsfx-api-key YOUR_ELEVENLABS_KEY
    --quiet
```

### Custom Service Integration
If you wish to use other services, you can implement your own using the following protocol:
```python
class VisionLMService(Protocol):
    def generate_audio_sources(self, prompt: str, images: list, api_key: str | None = None) -> dict:
        ...
class TTSFXService(Protocol):
    def generate_sound_effect(self, text_prompt: str, duration_seconds: float, prompt_influence: float, api_key: str = None) -> BytesIO:
        ...
```

## Importing in your own projects
```python
from auto_foley import run_auto_foley as af
output_path = af.add_audio_to_video("input_video.mp4")
```
Or with the optional extra parameters.
```python
from auto_foley import run_auto_foley as af
output_path = af.add_audio_to_video(
    input_video_path="input_video.mp4",
    output_video_path="output.mp4",          # Name and path for output
    frame_interval=25,                       # Extract one sample every 25 frames
    downscale_to_max_side=512,               # Downscale frames for lower vision LM input token cost
    prompt_instruction="Focus on the robot", # Add an instruction prompt which will be sent to the vision LM
    vision_lm_api_key="your-openai-api-key", # Pass API Key, for when environment variables are not set
    ttsfx_api_key="your-elevenlabs-api-key", # Pass API Key, for when environment variables are not set
    quiet=True                               # Suppress progress output and warnings
)
```
Instances of your custom service implementations can be passed to the `add_audio_to_video` function to overwrite the default services with.
```python
from auto_foley import run_auto_foley as af
vision_lm_service = YourCustomVisionLMClass()
ttsfx_Service = YourCustomTTSFXClass()
output = af.add_audio_to_video(
    input_video_path="input.mp4",
    overwrite_vision_lm=vision_lm_service,
    overwrite_ttsfx=ttsfx_Service
)
```

## Best Practices

1. **Video Length**: Works best with short clips (3-10 seconds). Longer videos may become unstable
2. **Frame Interval**: Defaults 1 sample per second, only use smaller intervals for fast-moving scenes
3. **Prompt Instructions**: Use this to give context or to exclude certain noises you don't want
4. **Resolution**: Lower resolutions process faster - downscale if full resolution isn't needed

## Limitations

- Maximum video duration recommended: 20 seconds
- Supported video format: mp4
- ElevenLabs sound effect duration must be between 0.5 and 22.0 seconds
