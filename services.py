import os
import sys
from io import BytesIO
from typing import Protocol, Optional

class VisionLMService(Protocol):
    def calculate_video_input_cost(self, width: int, height: int, sample_count: int) -> str:
        ...
    def generate_audio_sources(self, prompt: str, images: list, api_key: str | None = None) -> dict:
        ...

class TTSFXService(Protocol):
    def generate_sound_effect(self, text_prompt: str, duration_seconds: float, prompt_influence: float, api_key: str = None) -> BytesIO:
        ...

if __name__ == "__main__" or 'auto_foley' not in sys.modules:
    import vision_lm_chatgpt
    import ttsfx_elevenlabs
else:
    from auto_foley import vision_lm_chatgpt
    from auto_foley import ttsfx_elevenlabs

# Module-level service instances
_current_vision_lm: VisionLMService = vision_lm_chatgpt.ChatGPTVisionLM()
_current_ttsfx: TTSFXService = ttsfx_elevenlabs.ElevenLabsTTSFX()

def set_default_vision_lm(service: VisionLMService) -> None:
    """
    Sets the default global vision language model service for all functions to use.

    Args:
        service: Object instance that implements the VisionLMService protocol
    """
    global _current_vision_lm
    _current_vision_lm = service

def set_default_ttsfx(service: TTSFXService) -> None:
    """
    Sets the default global text-to-sound-effect service for all functions to use.

    Args:
        service: Object instance that implements the TTSFXService protocol
    """
    global _current_ttsfx
    _current_ttsfx = service

def get_services(vision_lm: Optional[VisionLMService] = None, ttsfx: Optional[TTSFXService] = None) -> tuple[VisionLMService, TTSFXService]:
    """
    Gets the service instances to use, either from provided arguments or falling back to global defaults.

    Args:
        vision_lm (optional): The vision language model service instance to use. 
            If None, uses the global default. Defaults to None
        ttsfx (optional): The text-to-sound-effect service instance to use.
            If None, uses the global default. Defaults to None

    Returns:
        tuple: Contains (vision_lm_service, ttsfx_service)
            vision_lm_service: VisionLMService instance to use
            ttsfx_service: TTSFXService instance to use
    """
    return (
        vision_lm if vision_lm is not None else _current_vision_lm,
        ttsfx if ttsfx is not None else _current_ttsfx
    )

def get_api_keys(vision_lm_api_key: Optional[str] = None, ttsfx_api_key: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """
    Gets the API keys to use, either from provided arguments or falling back to environment variables.
    
    Args:
        vision_lm_api_key (optional): The vision language model API key to use.
            If None or empty string, uses the AUTO_FOLEY_DEFAULT_VISION_LM_API_KEY environment variable.
            Defaults to None
        ttsfx_api_key (optional): The text-to-sound-effect API key to use.
            If None or empty string, uses the AUTO_FOLEY_DEFAULT_TTSFX_API_KEY environment variable.
            Defaults to None
            
    Returns:
        tuple: Contains (vision_lm_api_key, ttsfx_api_key)
            vision_lm_api_key: API key to use for vision language model service
            ttsfx_api_key: API key to use for text-to-sound-effect service
    """
    return (
        vision_lm_api_key if vision_lm_api_key else os.getenv('AUTO_FOLEY_DEFAULT_VISION_LM_API_KEY'),
        ttsfx_api_key if ttsfx_api_key else os.getenv('AUTO_FOLEY_DEFAULT_TTSFX_API_KEY')
    )
