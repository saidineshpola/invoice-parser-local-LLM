"""An example showing how to use vLLM to serve multimodal models 
and run online serving with OpenAI client.

Launch the vLLM server with the following command:

(single image inference with Llava)
vllm serve Qwen/Qwen2-VL-7B-Instruct

(multi-image inference with Phi-3.5-vision-instruct)
vllm serve microsoft/Phi-3.5-vision-instruct --task generate \
    --trust-remote-code --max-model-len 4096 --limit-mm-per-prompt image=2
"""

import os
import logging
from rich.logging import RichHandler
import base64
from typing import List
from pdf2image import convert_from_path

import requests
from openai import OpenAI

from vllm.utils import FlexibleArgumentParser


def setup_logger(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )


setup_logger(debug=False)
# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


def encode_base64_content_from_path(file_path: str) -> str:
    """Encode a content retrieved from a local file path to base64 format."""
    with open(file_path, "rb") as file:
        result = base64.b64encode(file.read()).decode("utf-8")
    return result


def run_text_only() -> None:
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "What's the capital of France?"}],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion.choices[0].message.content
    logging.info("Chat completion output: %s", result)


# Single-image input inference
def run_single_image(image_path: str) -> None:
    image_base64 = encode_base64_content_from_path(image_path)
    chat_completion_from_base64 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all the information present in the image into stuctured format.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_base64.choices[0].message.content
    logging.info("Chat completion output : %s", result)
    return result


# Multi-image input inference
def run_multi_image(image_paths: List[str]) -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What are present in these images?"}],
        }
    ]
    for image_path in image_paths:
        image_base64 = encode_base64_content_from_path(image_path)
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            }
        )

    chat_completion_from_base64 = client.chat.completions.create(
        messages=messages,
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_base64.choices[0].message.content
    logging.info("Chat completion output from base64 encoded images: %s", result)


# PDF input inference
def run_pdf(pdf_path: str) -> None:
    images = convert_from_path(pdf_path)
    os.makedirs("./tmp", exist_ok=True)
    for i, image in enumerate(images):
        image_path = f"./tmp/page_{i}.jpg"
        image.save(image_path, "JPEG")
        result = run_single_image(image_path)
        os.remove(image_path)
        break


example_function_map = {
    "text-only": run_text_only,
    "single-image": run_single_image,
    "multi-image": run_multi_image,
    "pdf": run_pdf,
}


def main(args) -> None:
    chat_type = args.chat_type
    if chat_type == "single-image":
        example_function_map[chat_type](args.image_path)
    elif chat_type == "multi-image":
        example_function_map[chat_type](args.image_paths)
    elif chat_type == "pdf":
        example_function_map[chat_type](args.pdf_path)
    else:
        example_function_map[chat_type]()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Demo on using OpenAI client for online serving with "
        "multimodal language models served with vLLM."
    )
    parser.add_argument(
        "--chat-type",
        "-c",
        type=str,
        default="pdf",
        choices=list(example_function_map.keys()),
        help="Conversation type with multimodal data.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        help="Path to the image file for single-image inference.",
    )
    parser.add_argument(
        "--image-paths",
        type=str,
        nargs="+",
        help="Paths to the image files for multi-image inference.",
    )
    parser.add_argument(
        "--pdf-path",
        type=str,
        help="Path to the PDF file for PDF inference.",
    )
    args = parser.parse_args()
    main(args)

