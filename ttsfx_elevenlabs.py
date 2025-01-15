from elevenlabs.client import ElevenLabs
from io import BytesIO

class ElevenLabsTTSFX:
    def generate_sound_effect(self, text_prompt: str, duration_seconds: float, prompt_influence: float = 0.3, api_key: str = None) -> BytesIO:
        """
        Generates a sound effect based on a text prompt using ElevenLabs API.
       
        Args:
            text_prompt: Text description of the desired sound effect
            duration_seconds: Length of the sound effect in seconds
            prompt_influence: How strongly the prompt influences the generated sound (default: 0.3)
            api_key: Optional API key for ElevenLabs authentication
           
        Returns:
            BytesIO: Audio data as a bytes stream
               
        Raises:
            ValueError: If text_prompt is empty or duration_seconds is invalid
            RuntimeError: If there's an error during sound effect generation
        """
        if not text_prompt:
            raise ValueError("Text prompt cannot be empty")
        # Other values will cause exceptions to be thrown
        if duration_seconds < 0.5:
            duration_seconds = 0.5
        if duration_seconds > 22.0:
            duration_seconds = 22.0
        try:
            elevenlabs_client = ElevenLabs(api_key=api_key)
            response = elevenlabs_client.text_to_sound_effects.convert(
                text=text_prompt,
                duration_seconds=duration_seconds,
                prompt_influence=prompt_influence,
            )
            # Collect the audio data into a BytesIO object
            audio_data = BytesIO()
            for chunk in response:
                audio_data.write(chunk)
            audio_data.seek(0)
            return audio_data
           
        except Exception as e:
            raise RuntimeError(f"{str(e)}")
    