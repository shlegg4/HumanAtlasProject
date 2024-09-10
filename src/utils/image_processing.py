import requests
from PIL import Image
import io
import numpy as np
from .logging import log_message

def download_image(image_url: str) -> Image.Image:
    """
    Downloads an image from the given URL and converts it into a PIL Image object.

    :param image_url: The URL of the image to download.
    :return: The downloaded image as a PIL Image object.
    """
    try:
        # Send a GET request to fetch the image content
        image_response = requests.get(url=image_url)
        log_message('info', f'{image_response=}')
        image_response.raise_for_status()  # Check if the request was successful
        
        # Convert the content to a PIL Image object
        image = Image.open(io.BytesIO(image_response.content))

        return image
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image: {e}")
        return None


def crop_image(image: Image.Image, boundary: list) -> Image.Image:
    """
    Crops the given PIL Image object using the specified boundary.

    :param image: The input image as a PIL Image object.
    :param boundary: A tuple (left, top, right, bottom) specifying the crop box.
    :return: The cropped image as a PIL Image object.
    """
    try:
        # Cropping the image using the provided boundary
        log_message('info', f'{image=} {tuple(boundary)=}')
        cropped_image = image.crop(tuple(boundary))
        return cropped_image
    except Exception as e:
        print(f"Failed to crop image: {e}")
        return None
