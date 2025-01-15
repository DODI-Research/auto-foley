from moviepy.audio.AudioClip import AudioArrayClip
from pydub import AudioSegment
import moviepy.editor as mpe
import numpy as np

def combine_video_and_audio(input_video_path: str, output_video_path: str, audio_sources: list) -> str:
    """
    Combines audio sources with a video file to create a new video with audio.
    
    Args:
        input_video_path: Path to the video file
        output_video_path: Path for the output video to be saved to
        audio_sources: List of audio source dictionaries containing timing and filepath information
        
    Returns:
        str: Path to the output video file with combined audio
            
    Raises:
        ValueError: If video path or audio sources are invalid or missing
        FileNotFoundError: If the video file cannot be accessed
        OSError: If there are issues with file operations in the output directory
        RuntimeError: If there's an error during the video composition process
    """
    if not input_video_path:
        raise ValueError("No video file provided")
    if not audio_sources:
        raise ValueError("No audio sources provided")
        
    try:
        video_clip = mpe.VideoFileClip(input_video_path)
        original_duration = video_clip.duration
        fps = video_clip.fps
        audio_clips = []

        for idx, audio_source in enumerate(audio_sources):
            audio_path = audio_source.get("AudioPath")
            start_frame_index = audio_source.get("StartFrameIndex", 0)
            volume = audio_source.get("Volume", 1.0)
            start_time = start_frame_index / fps # Convert start_frame_index to start_time in seconds

            if audio_path is None:
                continue  # Skip if no audio path

            # Convert the audio file to numpy array
            sample_rate, samples = convert_audio_file_to_numpy(audio_path)

            if samples is None:
                continue  # Skip if samples are None

            samples = np.array(samples)
            if samples.ndim == 0: # If samples is scalar, expand its dimensions
                samples = np.expand_dims(samples, axis=0)
            if samples.ndim == 1: # If samples is one-dimensional, reshape to (n_samples, 1)
                samples = samples.reshape(-1, 1)

            # Apply volume adjustment
            samples *= volume
            duration = len(samples) / sample_rate

            # Adjust duration if audio extends beyond video duration
            audio_end_time = start_time + duration
            if start_time >= original_duration:
                continue # Skip if audio clip starts after the video ends
            if audio_end_time > original_duration:
                # Adjust the duration to not exceed video duration
                duration = original_duration - start_time
                # Calculate the number of samples to keep
                num_samples = int(duration * sample_rate)
                samples = samples[:num_samples]

            audio_clip = AudioArrayClip(samples, fps=sample_rate).set_start(start_time).set_duration(duration)
            audio_clips.append(audio_clip)

        # Composite all audio clips
        if audio_clips:
            composite_audio = mpe.CompositeAudioClip(audio_clips).set_duration(original_duration)
            # Set the video's audio to the composite audio
            video_clip = video_clip.set_audio(composite_audio)
        else:
            # Ensure video has no audio if there are no audio clips
            video_clip = video_clip.without_audio()

        video_clip = video_clip.subclip(0, original_duration) # Trim the video to its original duration to prevent any extension
        video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", temp_audiofile=None, remove_temp=True, verbose=False, logger=None)

        video_clip.close()
        if audio_clips:
            for clip in audio_clips:
                clip.close()
            composite_audio.close()

        return output_video_path
    except Exception as e:
        raise RuntimeError(str(e))

def convert_audio_file_to_numpy(audio_filepath: str) -> tuple[int, np.ndarray]:
    """
    Converts an audio file to a numpy array with normalized samples.
    
    Args:
        audio_filepath: Path to the audio file to convert
        
    Returns:
        tuple: Contains (sample_rate, samples)
            - sample_rate (int): The sample rate of the audio
            - samples (np.ndarray): Normalized audio samples in range [-1, 1]
            
    Raises:
        FileNotFoundError: If the audio file cannot be accessed
        RuntimeError: If there's an error processing the audio file
    """
    try:
        audio_segment = AudioSegment.from_file(audio_filepath)
        sample_width = audio_segment.sample_width  # in bytes
        max_val = float(2 ** (8 * sample_width - 1))
        # Normalize samples to [-1, 1]
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / max_val
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2))
        sample_rate = audio_segment.frame_rate
        return (sample_rate, samples)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(str(e))
