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

SYS_MSG = r"""You are an evaluator for an open-ended benchmark converted from multiple-choice questions. 
Your job is to decide if the model’s answer is correct compared to the ground-truth. 
Follow the rules strictly, and do not invent new facts or regions.

### SCORING
- 1 = fully correct (semantically equivalent, exact match, or within tolerance / correct region)
- 0.5 = partially correct (overlaps with the truth but incomplete, approximate, or slightly outside tolerance/region)
- 0 = incorrect or irrelevant

### RULES

1. **Categorical answers (discrete labels, including spatial regions like "Top Left", "Bottom Right", land types, vegetation classes, etc.):**
   - Score 1 if the prediction matches the ground-truth label or a clear synonym.
   - Score 0.5 if the prediction is vague but overlaps the correct meaning.
   - Score 0 otherwise.

2. **Yes/No answers with additional detail:**
   - Score 1 only if both the Yes/No part and the required detail are correct.
   - Score 0.5 if the Yes/No part is correct but the detail is wrong.
   - Score 0 if the Yes/No part is wrong.

3. **Numeric answers (single values, counts, ranges, coordinates, bounding boxes, measurements):**
   - General numbers:
     - If GT < 100 → allow ±10% relative error.
     - If GT ≥ 100 → allow ±5% relative error.
     - If |GT| < 1 → use absolute tolerance ±0.1.
   - Numeric ranges:
     - Score 1 if the predicted value lies inside the reference range, or if predicted and reference ranges overlap substantially.
     - Score 0.5 if adjacent/nearby but not overlapping.
     - Score 0 if far outside.
   - Vectors (coordinates, bounding boxes, multi-value predictions):
     - Apply the same numeric tolerance to each element.
     - Score 1 if all or almost all elements are within tolerance.
     - Score 0.5 if partially within tolerance.
     - Score 0 otherwise.

4. **Geographic region answers (when GT refers to an ecological or geographic region rather than a point):**
   - Score 1 if the prediction is inside the same region as the ground-truth.
   - Score 0.5 if the prediction is close to or adjacent to that region.
   - Score 0 if the prediction refers to a clearly different region.
   - Do not invent new regions or boundaries. Only rely on widely known geography.

5. **Stages, phases, or temporal states:**
   - Score 1 if the prediction matches the ground-truth or is an obvious synonym (e.g., "Phase of initiation" ≈ "Initiation Phase").
   - Score 0.5 if semantically related but not exact (e.g., neighboring phase).
   - Score 0 otherwise.

---

### OUTPUT FORMAT
You must output a JSON object with two keys:
- `"reason"`: a short explanation of how the score was determined
- `"score"`: the numeric score (0, 0.5, or 1)

Example format:
{"reason": "<explanation>", "score": <0/0.5/1>}

---

### ICL EXAMPLE

Question: What phase is the event in at frame 4?  
Ground-truth Answer(s): Initiation Phase  
Model Answer: Phase of initiation  

Expected Output:
{"reason": "Model answer is a clear synonym of the ground-truth (Initiation Phase).", "score": 1}

---

### NOW EVALUATE

"""


@tenacity.retry(stop=tenacity.stop_after_attempt(6))
def call_openai_api(messages: Sequence[dict], client: openai.OpenAI, kwargs: dict):
    try:
        response = client.chat.completions.create(
            model=kwargs["model"],
            messages=messages,
            temperature=kwargs["temperature"],
            timeout=kwargs.get("timeout", 60),
            # max_tokens=cfg.openai.max_tokens,
            max_completion_tokens=kwargs["max_tokens"],
        )
    except Exception as e:
        print(f"API call failed: {e}. Retrying...")
        raise e
    text = response.choices[0].message.content.strip()
    text = text.strip("```").strip("```json")
    _ = json.loads(text)
    # reasoning = response.choices[0].message.reasoning_content.strip()
    return text, response


def process_item(item, kwargs):
    client = openai.OpenAI(
        api_key=omniearth.API_KEY,
        base_url=omniearth.API_BASE,
    )
    # client = OpenAI(
    #     api_key="sk-d9Ba2k5HnVJpmNz27FIuWqYOp10R4rMRgfi1xf1DczOY4RxQ",
    #     base_url="http://35.220.164.252:3888/v1",
    # )
    messages = []
    messages.append({"role": "system", "content": SYS_MSG})
    question = item["Text"]
    gt_answer = item["Ground Truth"]
    pred_answer = item["model_response"]
    prompt = f"Question: {question}\nGround-truth Answer(s): {gt_answer}\nModel Answer: {pred_answer}"
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
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
    parser.add_argument("--task-base", type=str)
    parser.add_argument("--task-path", "-t", type=str, required=True)
    parser.add_argument("--image-base", type=str, default=r"G:\dataset\OmniEarth\raw")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--workers", "-x", type=int, default=1)
    parser.add_argument("--max", type=int, required=True, default=-1)
    args = parser.parse_args()
    kwargs = args.__dict__
    kwargs = {
        "temperature": 0,
        "max_tokens": 1024,
        "timeout": 60,
    } | args.__dict__
    all_tasks = omniearth.utils.load_json(args.task_path)
    for task in all_tasks:
        jf = Path(args.task_base) / f"{task}.json"
        print(f"Processing file: {jf}")
        log_fname = Path("results") / args.model / f"{task}.json"
        log_fname.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_fname, "a+", buffering=1, encoding="utf-8")
        results = omniearth.utils.load_json(log_fname, is_jsonl=True)
        all_ids = {r["Question_id"] for r in results}
        data = omniearth.utils.load_json(jf, is_jsonl=True)
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
                item["scoring"] = output
                log_file.write(json.dumps(item, ensure_ascii=False) + "\n")
        log_file.close()
