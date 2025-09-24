import base64
import io
import json
import math
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Sequence
import orjson
from PIL import Image
from tqdm import tqdm
import omniearth
import tenacity
import openai

Image.MAX_IMAGE_PIXELS = 10_0000_0000


def encode_image(images: list, image_base: str | Path, limit=None) -> list:
    base64_strs = []
    for image in images:
        if type(image) is Image.Image:
            pass
        else:
            image = Image.open(Path(image_base) / image)
        image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        file_size = buffered.getbuffer().nbytes
        if limit is not None:  # 限制质量在 10~95 之间
            threshold = file_size / (limit * 1000_000)
            if threshold > 1:
                buffered = io.BytesIO()  # 重新创建 buffer 以清空之前的内容
                image.save(
                    buffered,
                    format="JPEG",
                    quality=min(100, int(95 / math.ceil(threshold))),
                )
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_strs.append(base64_str)
    return base64_strs


@tenacity.retry(stop=tenacity.stop_after_attempt(6))
def call_openai_api(messages: Sequence[dict], client: openai.OpenAI, kwargs: dict):
    response = client.chat.completions.create(
        model=kwargs["model"],
        messages=messages,
        temperature=kwargs["temperature"],
        timeout=kwargs.get("timeout", 60),
        # max_tokens=cfg.openai.max_tokens,
        max_completion_tokens=kwargs["max_tokens"],
    )
    text = response.choices[0].message.content.strip()
    # reasoning = response.choices[0].message.reasoning_content.strip()
    return text, response


def process_item(item, kwargs):
    model = kwargs["model"]
    client = openai.OpenAI(
        api_key=omniearth.API_KEY,
        base_url=omniearth.API_BASE,
    )
    if "intern" in model.lower():
        client = openai.OpenAI(
            api_key=omniearth.INTERN_API_KEY,
            base_url=omniearth.INTERN_API_BASE,
        )
    # client = OpenAI(
    #     api_key="sk-d9Ba2k5HnVJpmNz27FIuWqYOp10R4rMRgfi1xf1DczOY4RxQ",
    #     base_url="http://35.220.164.252:3888/v1",
    # )
    messages = []
    messages.append({"role": "system", "content": "You are a helpful assistant."})
    limit = 20
    if "claude" in model.lower():
        limit = 5
    elif "internvl" in model.lower():
        limit = 10
    if len(item["Images"]) > 1:
        limit = max(int(limit / len(item["Images"])), 1)
    base64_images = encode_image(item["Images"], limit=limit, image_base=kwargs["image_base"])
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Question: {item['Text']}"},
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                    for base64_image in base64_images
                ],
            ],
        }
    )
    try:
        text, response = call_openai_api(messages, client, kwargs)
        return True, item, text
    except Exception as e:
        print(f"Error processing item {item['Question_id']}: {e}")
        return False, item, str(e)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task-path", "-t", type=str, required=True)
    parser.add_argument("--task-base", type=str, default=r"G:\dataset\OmniEarth_iclr\jsons_open")
    parser.add_argument("--image-base", type=str, default=r"G:\dataset\OmniEarth\raw")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--workers", "-x", type=int, default=1)
    parser.add_argument("--max", type=int, required=True, default=-1)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    kwargs = args.__dict__
    kwargs = {
        "temperature": 0,
        "max_tokens": 2048,
        "timeout": args.timeout,
    } | args.__dict__
    all_tasks = omniearth.utils.load_json(args.task_path)
    for task in all_tasks:
        jf = Path(args.task_base) / f"{task}.json"
        print(f"Processing file: {jf}")
        log_fname = Path(args.model.replace("/", "_")) / f"{task}.json"
        log_fname.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_fname, "a+", buffering=1, encoding="utf-8")
        results = omniearth.utils.load_json(log_fname, is_jsonl=True)
        all_ids = {r["Question_id"] for r in results}
        data = omniearth.utils.load_json(jf)
        data = data[: args.max] if args.max > 0 else data
        MAX_WORKERS = args.workers
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futs = []
            for it in data:
                if it["Question_id"] in all_ids:
                    continue
                futs.append(executor.submit(process_item, it, kwargs))
            for success, item, output in omniearth.utils.finalize(
                futs,
                desc=f"Processing {jf.name}",
            ):
                if not success:
                    continue
                item["model_response"] = output
                log_file.write(json.dumps(item, ensure_ascii=False) + "\n")
        log_file.close()
