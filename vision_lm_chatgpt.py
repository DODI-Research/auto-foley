import base64
import io
import json
import math
import openai
from openai import OpenAI

class ChatGPTVisionLM:
    def calculate_video_input_cost(self, width: int, height: int, sample_count: int) -> str:
        """
        Calculates the estimated cost to process a video based on dimensions and number of samples.
        
        Args:
            width: Width of the video in pixels
            height: Height of the video in pixels
            sample_count: Number of frames to be processed
            
        Returns:
            str: Formatted string with calculated cost in USD with 6 decimal places
            
        Raises:
            ValueError: If width, height or sample_count are negative or zero
        """
        if width <= 0 or height <= 0 or sample_count <= 0:
            raise ValueError("Width, height and sample_count must be positive values")

        # Constants for gpt-4o-mini
        price_per_million_tokens = 0.15
        base_tokens = 2833
        tile_tokens = 5667
        tile_size = 512

        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)
        total_tiles = tiles_x * tiles_y
        total_tokens = base_tokens + (tile_tokens * total_tiles)
        cost_for_one_image = (total_tokens / 1000000) * price_per_million_tokens
        total_cost = cost_for_one_image * sample_count
        return f"${total_cost:.6f}"

    def generate_audio_sources(self, prompt: str, images: list, api_key: str | None = None) -> dict:
        """
        Generates a description of the video and a list of audio sources for the provided images using the OpenAI API.

        Args:
            prompt: The prompt to send to the model
            images: List of PIL.Image.Image objects to process
            api_key: Optional OpenAI API key. If None, will use environment variable
            
        Returns:
            dict: The parsed response containing:
                - VideoProperties (dict): Contains video description
                - AudioSources (list): List of audio source objects
                - AmbientAudioSources (list): List of ambient audio source objects
                
        Raises:
            ValueError: If images input is invalid
            RuntimeError: If there's an error calling the OpenAI API
            JSONDecodeError: If the API response cannot be parsed
        """
        gpt_model = "gpt-4o-mini"

        if not isinstance(images, list):
            images = [images]

        # Build the content list
        content_list = [{"type": "text", "text": prompt}]

        for image in images:
            # Encode the image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            content_list.append({
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "auto"
                },
            })

        # Construct messages
        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]

        # Define the tool (function) schema
        tool = {
                "type": "function",
                "function": {
                    "name": "analyze_video",
                    "description": "Provide a description of the video and extract audio sources from video frames.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "VideoProperties": {
                                "type": "object",
                                "properties": {
                                    "Description": {
                                        "type": "string",
                                        "description": "A general description of the entire video."
                                    }
                                },
                                "required": ["Description"],
                                "additionalProperties": False
                            },
                            "AudioSources": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "SourceSlugID": {
                                            "type": "string",
                                            "description": "A unique ID in the format {Subject}{Number}{Activity}, e.g., \"Bear1Roaring\"."
                                        },
                                        "SoundDescription": {
                                            "type": "string",
                                            "description": "Description of the sound, e.g., \"A person using a chainsaw against a tree.\""
                                        },
                                        "StartFrameIndex": {
                                            "type": "integer",
                                            "description": "The frame index where the sound starts."
                                        },
                                        "EndFrameIndex": {
                                            "type": "integer",
                                            "description": "The frame index where the sound ends."
                                        }
                                    },
                                    "required": ["SourceSlugID", "SoundDescription", "StartFrameIndex", "EndFrameIndex"],
                                    "additionalProperties": False
                                }
                            },
                            "AmbientAudioSources": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "SourceSlugID": {
                                            "type": "string",
                                            "description": "A unique ID in the format {Subject}{Number}{Activity}, e.g., \"Wind1Howling\"."
                                        },
                                        "SoundDescription": {
                                            "type": "string",
                                            "description": "Description of the ambient sound, e.g., \"An eerie wind howling through a ravine.\""
                                        },
                                        "StartFrameIndex": {
                                            "type": "integer",
                                            "description": "The frame index where the sound starts."
                                        },
                                        "EndFrameIndex": {
                                            "type": "integer",
                                            "description": "The frame index where the sound ends."
                                        }
                                    },
                                    "required": ["SourceSlugID", "SoundDescription", "StartFrameIndex", "EndFrameIndex"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["VideoProperties", "AudioSources", "AmbientAudioSources"],
                        "additionalProperties": False
                    }
                }
            }

        try:
            # Call the OpenAI API with the tool
            openai_client = OpenAI(api_key=api_key)
            response = openai_client.chat.completions.create(
                model=gpt_model,
                messages=messages,
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": "analyze_video"}},
                max_tokens=12288
            )

            # Access the choices and messages properly
            choice = response.choices[0]
            message = choice.message

            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                arguments = tool_call.function.arguments
                result_data = json.loads(arguments)
                return result_data
            else:
                raise RuntimeError("LLM did not provide the expected structured output.")

        except openai.OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error parsing API response: {str(e)}", e.doc, e.pos)
