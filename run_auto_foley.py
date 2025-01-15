import argparse
import base64
import cv2
import math
import os
import sys
import tempfile
from PIL import Image
from pydantic import BaseModel
from typing import Optional

if __name__ == "__main__" or 'auto_foley' not in sys.modules:
    import video_comping
    from services import (VisionLMService, TTSFXService, set_default_vision_lm, set_default_ttsfx, get_services, get_api_keys)
else:
    from auto_foley import video_comping
    from auto_foley.services import (VisionLMService, TTSFXService, set_default_vision_lm, set_default_ttsfx, get_services, get_api_keys)

class AudioSource(BaseModel):
    SourceSlugID: str
    SoundDescription: str
    StartFrameIndex: int
    EndFrameIndex: int
    Duration: float

AUDIO_SOURCES_KEY = 'AudioSources'
AMBIENT_AUDIO_SOURCES_KEY = 'AmbientAudioSources'
MINIMUM_AUDIO_DURATION = 0.5

# --- Low-level Helpers ---
def downscale_dimensions(width: int, height: int, max_side: int) -> tuple[int, int]:
    """
    Resize dimensions down while maintaining aspect ratio and respecting maximum side length.
    
    Args:
        width: Original width of the image/video
        height: Original height of the image/video
        max_side: Maximum allowed length for either dimension
        
    Returns:
        tuple: (new_width, new_height) maintaining original aspect ratio
            
    Raises:
        ValueError: If width, height, or max_side are less than or equal to 0
    """
    if width <= 0 or height <= 0 or max_side <= 0:
        raise ValueError("Width, height, and max_side must all be positive integers")   
    if width <= max_side and height <= max_side:
        return (width, height)
    
    aspect_ratio = width / height
    if width > height:
        new_width = max_side
        new_height = int(max_side / aspect_ratio)
    else:
        new_height = max_side
        new_width = int(max_side * aspect_ratio)
    return (new_width, new_height)

def calculate_duration(start_frame: int, end_frame: int, fps: int, frame_count: int) -> tuple[float, int, int]:
    """
    Calculate duration in seconds based on frame indices and FPS.
    
    Args:
        start_frame: Starting frame index
        end_frame: Ending frame index
        fps: Frames per second of the video
        frame_count: Total number of frames in the video
        
    Returns:
        tuple: A tuple containing:
            float: Duration in seconds (rounded to 3 decimal places)
            int: Adjusted start frame index
            int: Adjusted end frame index
            
    Raises:
        ValueError: If fps is less than or equal to 0
        ValueError: If frame_count is less than or equal to 0
    """
    # Input validation
    if fps <= 0:
        raise ValueError("FPS must be greater than 0")
    if frame_count <= 0:
        raise ValueError("Frame count must be greater than 0")    
    
    start = max(0, start_frame)
    end = min(frame_count - 1, end_frame) if end_frame is not None else frame_count - 1
    if end <= start:
        end = start + fps

    duration_frames = end - start
    duration_seconds = duration_frames / fps
    # If duration is less than minimum, adjust frames
    if duration_seconds < MINIMUM_AUDIO_DURATION:
        frames_needed = int(MINIMUM_AUDIO_DURATION * fps) - duration_frames
        # First try to extend end frame
        target_end = end + frames_needed
        if target_end <= frame_count - 1:
            end = target_end
        else:
            # Can't extend end frame, need to adjust start frame instead
            end = frame_count - 1
            remaining_frames_needed = int(MINIMUM_AUDIO_DURATION * fps) - (end - start)
            potential_start = start - remaining_frames_needed
            start = max(0, potential_start)

    final_duration_frames = end - start
    final_duration = final_duration_frames / fps
    return round(final_duration, 3), start, end

def combine_all_audio_sources(all_audio_sources: dict) -> list:
    """
    Helper function to convert format of audio sources into a single combined list.
    
    Args:
        all_audio_sources: Dictionary containing AudioSources and AmbientAudioSources lists
            Expected format: {'AudioSources': [], 'AmbientAudioSources': []}
        
    Returns:
        list: Combined list of all audio sources from both categories
    """
    audio_sources = all_audio_sources.get(AUDIO_SOURCES_KEY, [])
    ambient_audio_sources = all_audio_sources.get(AMBIENT_AUDIO_SOURCES_KEY, [])
    return audio_sources + ambient_audio_sources

# --- Video ---
def get_video_path(video) -> str:
    """
    Converts video input into a file path, creating a temporary file if needed.
    
    Args:
        video: Video input that can be a file path string, a dictionary with video data, or a dictionary with a video name
              
    Returns:
        str: Path to the video file
        
    Raises:
        ValueError: If no video is provided or if video format is invalid
        RuntimeError: If video data cannot be processed
    """
    if not video:
        raise ValueError("No video file provided.")
    if isinstance(video, str):
        return video
    elif isinstance(video, dict):
        video_data = video.get('data')
        video_name = video.get('name')
        if video_name:
            return video_name
        else:
            try:
                # Save video data to temporary file
                header, encoded = video_data.split(',', 1)
                file_ext = header.split(';')[0].split('/')[1]
                video_bytes = base64.b64decode(encoded)

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.' + file_ext)
                temp_file.write(video_bytes)
                temp_file.close()
                return temp_file.name
            except Exception as e:
                raise RuntimeError(f"Failed to process video data: {e}")
    else:
        raise ValueError("Invalid video format")

def get_video_info(video_path: str) -> dict:
    """
    Extracts information about a video.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        A dictionary containing video metadata:
            {
                "FilePath": str,
                "Width": int,
                "Height": int,
                "Duration": float,
                "FrameCount": int,
                "FrameRate": float,
                "FrameInterval": float
            }
            
    Raises:
        ValueError: If video_path is None or empty
        FileNotFoundError: If the video file cannot be opened
        RuntimeError: If there's an error processing the video
    """
    if not video_path:
        raise ValueError("Video file path was not given")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open the video file at '{video_path}'. Please verify the file path and format.")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    except Exception as e:
        raise RuntimeError(f"{e}")
    return {
        "FilePath": video_path,
        "Width": width,
        "Height": height,
        "Duration": total_frames / fps if fps != 0 else 0,
        "FrameCount": total_frames,
        "FrameRate": fps,
        "FrameInterval": fps
    }

def extract_frames(video_path: str, frame_interval: int, target_width: int = None, target_height: int = None) -> list:
    """
    Extracts frames from the video at the specified interval and includes the last frame.
    Optionally downscales the frames to the target resolution.
    
    Args:
        video_path (str): Path to the video file
        frame_interval (int): Interval between extracted frames
        target_width (int, optional): Desired width of output frames. If None, original width is used
        target_height (int, optional): Desired height of output frames. If None, original height is used
        
    Returns:
        list: List of tuples (frame_index, PIL Image)
            
    Raises:
        FileNotFoundError: If the video file cannot be opened
        ValueError: If video contains no frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError("Could not open video.")  
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError("Video contains no frames.")

    # Calculate target dimensions while maintaining aspect ratio if only one dimension is specified
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if target_width is not None and target_height is None:
        scale_factor = target_width / original_width
        target_height = int(original_height * scale_factor)
    elif target_height is not None and target_width is None:
        scale_factor = target_height / original_height
        target_width = int(original_width * scale_factor)

    need_resize = (target_width is not None and target_height is not None and (target_width != original_width or target_height != original_height))
    
    # Compute the frame indices to extract
    if frame_interval <= 0 or frame_interval >= total_frames:
        frame_indices = [0, total_frames - 1] if total_frames > 1 else [0]
    else:
        frame_indices = list(range(0, total_frames, frame_interval))
        if frame_indices[-1] != total_frames - 1:
            frame_indices.append(total_frames - 1)
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue  # Skip if frame could not be read
        if need_resize:
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append((frame_idx, pil_image))
    
    cap.release()
    return frames

def calculate_video_input_cost(width: int, height: int, sample_count: int, overwrite_vision_lm: Optional[VisionLMService] = None) -> float:
    """
    Calculates the input cost for video processing based on dimensions and sample count.
    
    Args:
        width: Width of the video in pixels
        height: Height of the video in pixels
        sample_count: Number of samples/frames being processed
        overwrite_vision_lm: Optional custom object instance that implements the VisionLMService protocol to use instead of the default (ChatGPT)

    Returns:
        float: The calculated input cost for video processing
        
    Raises:
        ValueError: If width, height, or sample_count are not positive numbers
        RuntimeError: If there's an error during cost calculation
    """
    vision_lm, _ = get_services(vision_lm=overwrite_vision_lm)
    if width <= 0 or height <= 0 or sample_count <= 0:
        raise ValueError("Width, height, and sample count must be positive numbers")
    try:
        return vision_lm.calculate_video_input_cost(width, height, sample_count)
    except Exception as e:
        raise RuntimeError({str(e)})

# --- Audio ---
def validate_audio_sources(audio_sources: list, fps: int, frame_count: int) -> list:
    """
    Helper function to validate a list of audio sources while calculating durations 
    and adjusting frame indices to ensure minimum audio duration.
    
    Args:
        audio_sources: List of dictionaries containing audio source information
        fps: Frames per second of the video
        frame_count: Total number of frames in the video
        
    Returns:
        List of validated audio source dictionaries with adjusted durations and frame indices
        
    Raises:
        ValueError: If audio_sources is None or empty
        TypeError: If any audio source item is missing required fields
        ValueError: If frame indices are invalid or if fps is 0
    """
    if not audio_sources:
        raise ValueError("Audio sources list cannot be empty")   
    if fps <= 0:
        raise ValueError("FPS must be greater than 0")
    
    validated_sources = []
    for audio_source in audio_sources:
        try:
            # Calculate audio duration and get adjusted frame indices
            duration, start_frame, end_frame = calculate_duration(
                audio_source.get("StartFrameIndex", 0), 
                audio_source.get("EndFrameIndex", frame_count - 1), 
                fps, 
                frame_count
            )
            audio_source["Duration"] = duration
            audio_source["StartFrameIndex"] = start_frame
            audio_source["EndFrameIndex"] = end_frame
            validated_audio_source = AudioSource(**audio_source)
            validated_sources.append(validated_audio_source.model_dump())
        except (KeyError, TypeError) as e:
            raise TypeError(f"Invalid audio source format: {str(e)}")           
    return validated_sources

def generate_audio(prompt: str, duration: float, ttsfx_api_key: str = None, overwrite_ttsfx: Optional[TTSFXService] = None) -> str:
    """
    Generates audio file for given prompt and duration.
    
    Args:
        prompt: Description of the sound to generate
        duration: Length of the audio in seconds
        ttsfx_api_key: API key for the text-to-speech service
        overwrite_ttsfx: Optional custom object instance that implements the TTSFXService protocol to use instead of the default (ElevenLabs)
        
    Returns:
        str: Path to the generated audio file
        
    Raises:
        RuntimeError: If there's an error during audio generation
    """
    _, ttsfx = get_services(ttsfx=overwrite_ttsfx)
    _, ttsfx_api_key = get_api_keys(ttsfx_api_key=ttsfx_api_key)
    try:
        audio_data = ttsfx.generate_sound_effect(prompt, duration, api_key=ttsfx_api_key)
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_audio_file.write(audio_data.read())
        temp_audio_file.close()
        return temp_audio_file.name
    except Exception as e:
        raise RuntimeError(str(e))

def generate_audio_for_audio_source(audio_source: dict, ttsfx_api_key: str = None, overwrite_ttsfx: Optional[TTSFXService] = None) -> dict:
    """
    Generates audio file for single audio source with description.
    
    Args:
        audio_source: Dictionary containing sound description and duration
        ttsfx_api_key: API key for the text-to-speech service
        overwrite_ttsfx: Optional custom object instance that implements the TTSFXService protocol to use instead of the default (ElevenLabs)
        
    Returns:
        dict: Audio source dictionary with added AudioPath and Volume fields
    """
    sound_description = audio_source.get("SoundDescription", "")
    duration = audio_source.get("Duration")
    audio_source["AudioPath"] = generate_audio(sound_description, duration, ttsfx_api_key, overwrite_ttsfx)
    audio_source["Volume"] = 1.0
    return audio_source

def generate_all_audio(all_audio_sources: dict, ttsfx_api_key: str = None, overwrite_ttsfx: Optional[TTSFXService] = None) -> dict:
    """
    Generates audio files for audio sources with descriptions.
    
    Args:
        all_audio_sources: Dictionary containing AudioSources and AmbientAudioSources lists
        ttsfx_api_key: API key for the text-to-speech service
        overwrite_ttsfx: Optional custom object instance that implements the TTSFXService protocol to use instead of the default (ElevenLabs)
        
    Returns:
        dict: Audio sources dictionary with generated audio paths
        
    Raises:
        ValueError: If audio sources dictionary is None
        RuntimeError: If there's an error during audio generation
    """
    if all_audio_sources is None:
        raise ValueError("missing audio sources.")
    try:
        for audio_source in all_audio_sources[AUDIO_SOURCES_KEY]:
            audio_source = generate_audio_for_audio_source(audio_source, ttsfx_api_key, overwrite_ttsfx)
        for audio_source in all_audio_sources[AMBIENT_AUDIO_SOURCES_KEY]:
            audio_source = generate_audio_for_audio_source(audio_source, ttsfx_api_key, overwrite_ttsfx)
        return all_audio_sources
    except Exception as e:
        raise RuntimeError(str(e))

# --- High-level Processing ---
def process_video(video, frame_interval=None, target_width=None, target_height=None, prompt_instruction=None, vision_lm_api_key=None, overwrite_vision_lm: Optional[VisionLMService] = None) -> tuple[dict, str]:
    """
    Processes the video, extracts frames at the specified interval, generates video description and audio sources.

    Args:
        video: Video input as either a file path string, or a dictionary containing video data
        frame_interval: Interval between extracted frames. Defaults to None
        target_width (optional): Target width for frame extraction. Defaults to None
        target_height (optional): Target height for frame extraction. Defaults to None
        prompt_instruction (optional): Custom instruction to include in the prompt. Defaults to None
        vision_lm_api_key (optional): API key for vision language model. Defaults to None
        overwrite_vision_lm (optional): Custom object instance that implements the VisionLMService protocol to use instead of the default (ChatGPT)

    Returns:
        tuple: Contains (audio_sources, video_description)
            audio_sources: Dictionary containing 'AudioSources' and 'AmbientAudioSources' lists
            video_description: String containing detailed description of the video

    Raises:
        ValueError: If video input is invalid or missing
        FileNotFoundError: If video file cannot be opened or accessed
        RuntimeError: If frame extraction fails, API returns an error, or audio source generation fails
        Exception: If data validation fails
    """
    # Add frame info
    video_path = get_video_path(video)
    cap = cv2.VideoCapture(video_path)
    fps = 25 # Default value
    frame_count = fps
    frame_info_prompt = ""

    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if frame_interval is None:
            frame_interval = math.ceil(fps) # Default to one sample per second
        seconds = frame_interval / fps
        frame_info_prompt += f"\nThe framerate of the video is {fps} FPS."
        frame_info_prompt += f"\nThe frame interval between each image is {frame_interval} frames."
        frame_info_prompt += f" Which means there's approximately {seconds:.2f} seconds between each image in real time."
        cap.release()

    # Extract frames
    frames = extract_frames(video_path, frame_interval, target_width, target_height)
    if not frames:
        raise RuntimeError("No frames extracted from video.")
    
    images = []
    frame_numbers = []
    for frame_idx, pil_image in frames:
        images.append(pil_image)
        frame_numbers.append(frame_idx)

    # Build prompt
    prompt = (
        "You will be given a list of images representing certain frames of a video that has no audio."
        "\nYou must provide a general description of the video and a list of audio sources you recognize"
        " in these frames, so that it can be used to create an audio file for each audio source in the video."
        
        "\n\nThe VideoProperties 'Description' property must include a very thorough and detailed description of everything that is visible in the frames."
        " Make sure not a single thing is missing, the context of this description will help define the audio sources."
        "\nThe VideoProperties 'Description' property may also include descriptions of abstract things,"
        " like the vibe, the medium (if it's animated, cartoony or recorded) or the time period,"
        " but only if it's relevant to the possible audio that could fit underneath the video."
        
        "\n\nThe AudioSource 'SoundDescription' property is implied to already be a description of audio so it does not contain words like \"The sound of ...\", etc."
        "\nThe AudioSource 'SoundDescription' property should only contain the direct description of the audio, and not any unnecessary visual descriptions like color."
        "\nThe AudioSource 'SoundDescription' property should be able to be understood without outside context. For example, you should never describe it as"
        " \"The sound of footsteps as the mayor is walking towards the lamp post.\". Because \"the mayor\" or \"the lamp post\" are unknown entities within this description. They're also irrelevant to the sound this would make."
        " The correct description for that example would have been: \"Footsteps of a man on a sidewalk.\""
        "\nThe AudioSource 'SoundDescription' property also does not know about the other properties, like the AudioSource 'Slug' property."
        " If the audio source is a bird, for example, the 'SoundDescription' property can't just be \"Wings flapping.\", but must still include the subject if it would be relevant for the sound,"
        " so a better description would be \"A bird flapping its wings.\""

        "\n\nAnother example of a very bad sound description, this is an example of things you need to avoid: \"A gray owl with a hat is making sounds on the walls of the castle, adding to the eerie vibe of the video.\""
        "\nThe correct way to describe that would be: \"An owl hooting in the distance.\""
        
        "\n\nLastly, you will also include a list of ambient audio sources."
        "\nThe ambient audio sources are similar to the normal audio sources, but are for elements that are either invisible in the frames or just not a single source of audio."
        "\nThe sounds from the ambient audio sources should not already exist in the normal audio sources. Because that would mean that sound would be duplicate in the end result."
        "\nTo create a good ambient audio source description, you must imagine what you could hear in the video that's not necessarily depicted."
        "\nExample: if a video depicts a squirrel playing in a forest, then you can imagine the forest making lots of noises that aren't directly visible,"
        " such as chirping birds or crickets. The ambient sound description will not include the noises the squirrel makes because these will be found in the normal audio sources.\n"
    )

    if frame_info_prompt is not None and frame_info_prompt.strip():
        prompt += frame_info_prompt
    if prompt_instruction is not None and prompt_instruction.strip():
        prompt += f"\n\nThe user included the following custom instruction for you, try to abide if possible:\n<user-instruction>\n{prompt_instruction}\n</user-instruction>"
    prompt += f"\nThis array shows the frame index (which frame of the video they represent), for each image you are about to see in order: {frame_numbers}"

    # Generate video description and audio sources
    try:
        vision_lm, _ = get_services(vision_lm=overwrite_vision_lm)
        vision_lm_api_key, _ = get_api_keys(vision_lm_api_key=vision_lm_api_key)
        result_data = vision_lm.generate_audio_sources(prompt, images, vision_lm_api_key)
    except Exception as e:
        raise RuntimeError(str(e))

    # Validate the returned data using the AudioSource model
    video_description = result_data['VideoProperties']['Description']
    audio_sources_data = result_data[AUDIO_SOURCES_KEY]
    ambient_audio_sources_data = result_data[AMBIENT_AUDIO_SOURCES_KEY]

    try:
        validated_audio_sources = validate_audio_sources(audio_sources_data, fps, frame_count)
        validated_ambient_audio_sources = validate_audio_sources(ambient_audio_sources_data, fps, frame_count)
        all_audio_sources = { AUDIO_SOURCES_KEY: validated_audio_sources, AMBIENT_AUDIO_SOURCES_KEY: validated_ambient_audio_sources }
        return all_audio_sources, video_description
    except Exception as e:
        raise Exception(str(e))

def combine_video_and_audio(audio_sources: list, input_video: str | dict, output_video_path: str = None) -> str:
    """
    Composes audio sources onto a video file, creating a new video with the combined audio.
    
    Args:
        audio_sources (list): List of audio source dictionaries containing audio paths and timing
        input_video (str | dict): Either a path to video file or a dictionary containing video data
        output_video_path (str): Path for the output video to be saved to, if none will be adjecant to the input_video path and add an '_output' suffix. Defaults to None
        
    Returns:
        str: Path to the output video file with composed audio
        
    Raises:
        ValueError: If video input is invalid or missing
        FileNotFoundError: If video file cannot be accessed
        RuntimeError: If error occurs during video composition or audio processing
        OSError: If there are file system errors during composition
    """
    input_video_path = get_video_path(input_video)
    if not input_video_path:
        raise ValueError("Invalid or missing video input")   
    try:
        if output_video_path is None:
            path_to_input_file_name, input_file_extension = os.path.splitext(input_video_path)
            output_video_path = path_to_input_file_name + "_output" + input_file_extension
        return video_comping.combine_video_and_audio(input_video_path, output_video_path, combine_all_audio_sources(audio_sources))
    except Exception as e:
        raise RuntimeError(str(e))
    
def add_audio_to_video(input_video_path, output_video_path=None, frame_interval=None, downscale_to_max_side=None, prompt_instruction=None, vision_lm_api_key=None, ttsfx_api_key: str = None, overwrite_vision_lm: Optional[VisionLMService] = None, overwrite_ttsfx: Optional[TTSFXService] = None, quiet: bool = False) -> str:
    """
    Processes the video, extracts frames at the specified interval, generates video description and audio sources, then generate audio from those descriptions and adds them back to the video.

    Args:
        video: Video input as either a file path string, or a dictionary containing video data
        frame_interval (optional): Interval between extracted frames. Defaults to None
        downscale_to_max_side (optional): Downscale samples to maximum side, if None will use 512px. Defaults to None
        prompt_instruction (optional): Custom instruction to include in the prompt. Defaults to None
        vision_lm_api_key (optional): API key for vision language model. Defaults to None
        overwrite_vision_lm (optional): Custom object instance that implements the VisionLMService protocol to use instead of the default (ChatGPT)
        quiet (optional): If True, suppresses progress output. Defaults to False

    Returns:
        str: Path to the output video file with composed audio

    Raises:
        ValueError: If video input is invalid or missing
        FileNotFoundError: If video file cannot be opened or accessed
        RuntimeError: If frame extraction fails, API returns an error, audio source generation fails, error occurs during video composition or audio processing
        Exception: If data validation fails
        OSError: If there are file system errors during composition
    """
    try:
        if downscale_to_max_side is None:
            downscale_to_max_side = 512
        input_video_info = get_video_info(input_video_path)
        target_width, target_height = downscale_dimensions(input_video_info['Width'], input_video_info['Height'], downscale_to_max_side)
        duration = input_video_info.get("Duration", 0.0)
        if duration > 20.0 and overwrite_ttsfx is None:
            if not quiet:
                print("WARNING: Input video is longer than 20 seconds. This process is works best for 3-10 second clips and becomes unstable with longer videos. Consider trimming your video into shorter segments.")

    except Exception as e:
        raise RuntimeError(f"Error determining downscaled sample resolution: {e}")

    try:
        if not quiet:
            print(f"Processing {input_video_path}...")
        all_audio_sources, _ = process_video(input_video_path, frame_interval, target_width, target_height, prompt_instruction, vision_lm_api_key, overwrite_vision_lm)
    except Exception as e:
        raise RuntimeError(f"Error during video processing: {e}")
    
    try:
        if not quiet:
            print(f"Generating audio...")
        all_audio_sources = generate_all_audio(all_audio_sources, ttsfx_api_key, overwrite_ttsfx)
    except Exception as e:
        raise RuntimeError(f"Error during audio generation: {e}")
    
    try:
        print(f"Adding generated audio to the video...")
        output = combine_video_and_audio(all_audio_sources, input_video_path, output_video_path)
        if not quiet:
            print(f"Output saved to: {output}")
        return output
    except Exception as e:
        raise RuntimeError(f"Error while combining video and audio: {e}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Auto-foley: Automatically add sound effects to a video without sound',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i', 
        required=True,
        help='Path to the input video file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output', '-o',
        help='Path for the output video file. If not specified, will append "_output" to the input filename'
    )
    
    parser.add_argument(
        '--frame-interval',
        type=int,
        help='Interval between processed frames'
    )
    
    parser.add_argument(
        '--downscale-to-max-side',
        type=int,
        help='Target max side for downscaling video samples'
    )
    
    parser.add_argument(
        '--prompt-instruction',
        help='Custom instruction for the video processing'
    )
    
    parser.add_argument(
        '--vision-lm-api-key',
        help='API key for the vision language model'
    )
    
    parser.add_argument(
        '--ttsfx-api-key',
        help='API key for the text-to-speech effects service'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    try:
        add_audio_to_video(
            input_video_path=args.input,
            output_video_path=args.output,
            frame_interval=args.frame_interval,
            downscale_to_max_side=args.downscale_to_max_side,
            prompt_instruction=args.prompt_instruction,
            vision_lm_api_key=args.vision_lm_api_key,
            ttsfx_api_key=args.ttsfx_api_key,
            quiet=args.quiet
        )
    except Exception as e:
        print(f"{str(e)}", file=sys.stderr)
        sys.exit(1)