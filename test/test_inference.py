import torch
from lit_upscaler_mock import LitUpscalerMock
from model.inference import Inference
from model.image_utils import ycbcr_tensor_to_pil


def validate_output_shape(inference: Inference, img_size: tuple[int, int]) -> None:
    input_tensor = torch.randn(1, 3, img_size[0], img_size[1])
    img = ycbcr_tensor_to_pil(input_tensor[0])
    img_out = inference.upscale(img)

    correct_size = (img.size[0] * inference.UPSCALE_FACTOR, img.size[1] * inference.UPSCALE_FACTOR)
    assert img_out.size == correct_size, f"Output for {img.size} should be {correct_size}"


def test_large_img_dimensions() -> None:
    mock_upscaler = LitUpscalerMock(upscale_factor=2)
    inference = Inference(model=mock_upscaler)
    validate_output_shape(inference, (1000, 1000))
    validate_output_shape(inference, (2000, 2000))
    validate_output_shape(inference, (1920, 1080))
    validate_output_shape(inference, (640, 480))
    validate_output_shape(inference, (320, 240))


def test_small_img_dimensions() -> None:
    mock_upscaler = LitUpscalerMock(upscale_factor=2)
    inference = Inference(model=mock_upscaler)
    validate_output_shape(inference, (100, 100))
    validate_output_shape(inference, (100, 100))
    validate_output_shape(inference, (200, 100))
    validate_output_shape(inference, (50, 50))
    validate_output_shape(inference, (50, 200))