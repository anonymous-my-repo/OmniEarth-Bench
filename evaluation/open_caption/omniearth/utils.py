from PIL import Image
from pathlib import Path
from collections.abc import Callable, Sequence
import json
import orjson
from tqdm.auto import tqdm
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    Future,
)
import base64
from io import BytesIO

Image.MAX_IMAGE_PIXELS = None


def load_json(file_path: Path, is_jsonl: bool = False):
    if is_jsonl:
        with open(file_path, "r", encoding="utf-8") as f:
            return [orjson.loads(line) for line in f]
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return orjson.loads(f.read())


def write_json(data: list, file_path: Path, is_jsonl: bool = False, **kwargs):
    if is_jsonl:
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, **kwargs)
            # f.write(orjson.dumps(data).decode("utf-8"))


def encode_img_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    buffered = BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def decode_base64_to_img(img_str: str) -> Image.Image:
    img_data = base64.b64decode(img_str)
    image = Image.open(BytesIO(img_data))
    return image


def finalize(futs: list[Future], desc: str = "Processing", disable_tqdm: bool = False) -> list:
    results = []
    for fut in tqdm(as_completed(futs), desc=desc, disable=disable_tqdm, total=len(futs)):
        results.append(fut.result())
    return results
