import re
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Sequence
from tqdm import tqdm
import omniearth
from loguru import logger as eval_logger
from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO


def load_json(fpath, is_json_line=False):
    res = []
    with open(fpath, "r", encoding="utf8") as f:
        if is_json_line:
            for line in f:
                res.append(json.loads(line))
        else:
            res = json.load(f)
    return res


def write_json(data, fpath, is_json_line=False):
    with open(fpath, "w", encoding="utf8") as f:
        if is_json_line:
            for item in tqdm(data):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)


COCO_METRICS = [
    "Bleu_1",
    "Bleu_2",
    "Bleu_3",
    "Bleu_4",
    "METEOR",
    "ROUGE_L",
    "CIDEr",
]  # , "SPICE"]


def compute_metrics(results, metric):
    scorers = [
        (Bleu(4), "Bleu_1"),
        (Bleu(4), "Bleu_2"),
        (Bleu(4), "Bleu_3"),
        (Bleu(4), "Bleu_4"),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]  # , (Spice(), "SPICE")]
    scorers_dict = {s[1]: s for s in scorers}

    stored_results = []
    # In order to make the coco eval tools to successfully create index
    # We need at least two dict in the dataset
    # 'annotation' and 'images'
    # 'annotation' exactly reproduce the original annotation
    # 'images' however only need the image id which is contained in the file name
    dataset = {"annotations": [], "images": []}
    idx = 0
    for result in results:
        stored_results.append({"image_id": int(result["image_id"]), "caption": result["pred"]})
        dataset["annotations"].append(
            {
                "image_id": int(result["image_id"]),
                "caption": result["answer"],
                "id": idx,
            }
        )
        dataset["images"].append({"id": result["image_id"]})

    coco = COCO()
    # Manually create index here
    coco.dataset = dataset
    coco.createIndex()

    coco_result = coco.loadRes(stored_results)
    coco_eval = COCOEvalCap(coco, coco_result)

    imgIds = coco_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
        res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]

    eval_logger.info("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    eval_logger.info(f"Computing {metric} scores...")

    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    # When metric is one of the Bleu, score will be a list
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    return score


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task-base", type=str)
    parser.add_argument("--task-path", "-t", type=str, required=True)
    args = parser.parse_args()
    kwargs = args.__dict__
    all_tasks = omniearth.utils.load_json(args.task_path)
    data = []
    # for jf in Path()
    for task in all_tasks:
        jf = Path(args.task_base) / f"{task}.json"
        data.extend(omniearth.utils.load_json(jf, is_jsonl=True))
    new_data = []
    for idx, it in enumerate(data):
        new_data.append(
            {
                "answer": re.sub(r"[#\*\n]", "", it["Ground Truth"]),
                "pred": re.sub(r"[#\*\n]", "", it["model_response"]),
                "image_id": idx,
                "id": idx,
            }
        )
    metrics = dict()
    for metric in COCO_METRICS:
        metrics[metric] = compute_metrics(new_data, metric)
    print(metrics)
    for metric in COCO_METRICS:
        print(f"{metric:<20}", metrics[metric])
