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

Image.MAX_IMAGE_PIXELS = 10_0000_0000


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task-base", type=str)
    parser.add_argument("--task-path", "-t", type=str, required=True)
    # parser.add_argument("--image-base", type=str, default=r"G:\dataset\OmniEarth\raw")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode")
    args = parser.parse_args()
    kwargs = args.__dict__
    print(kwargs)
    all_tasks = omniearth.utils.load_json(args.task_path)
    for task in all_tasks:
        jf = Path(args.task_base) / f"{task}.json"
    # all_tasks = omniearth.utils.load_json(args.task_path)
    # for task in Path(args.task_base).rglob("*.json"):
    #     jf = task
        # print(f"Processing file: {jf}")
        data = omniearth.utils.load_json(jf, is_jsonl=True)
        scores = []
        for it in data:
            scoring = orjson.loads(it["scoring"])
            scores.append(scoring["score"])
        if len(scores) == 0:
            print(f"{str(task):<120} ## {0} with 0 samples")

            continue
        score = sum(scores) / len(scores)
        if args.strict:
            cnt = sum(1 for s in scores if s >= 1)
            score = cnt / len(scores)
        # print(f"{task:<120} ## {cnt / len(data) * 100}")
        print(f"{str(task):<120} ## {score}")
