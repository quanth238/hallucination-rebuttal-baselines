"""Python 3 compatibility implementation of the official COCO CHAIR metric.

The metric semantics follow ``LisaAnne/Hallucination/utils/chair.py`` at the
commit recorded below.  This module intentionally keeps the original synonym
table, token normalization, double-word rules, and the union of instance and
human-caption annotations.  It does not use the third-party standalone pickle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import inflect
import nltk


_INFLECT = inflect.engine()


def singularize(word: str) -> str:
    """Return the inflect singular form, preserving already-singular tokens."""

    result = _INFLECT.singular_noun(word)
    return str(result) if result else word


OFFICIAL_REPOSITORY = "LisaAnne/Hallucination"
OFFICIAL_COMMIT = "6e4d33c4bedf6dfcd24c61bf527f40714717be47"
OFFICIAL_CHAIR_SHA256 = "b05f701ce6ec0125dd700ffb80ed58bfcf2686df8c9bd62f70169cb928de4627"
OFFICIAL_SYNONYMS_SHA256 = "b685537b36fe21b791c87151b96a991ceb582edff19f3da204dcc7a7905502f2"
ANNOTATION_FILES = (
    "instances_train2014.json",
    "instances_val2014.json",
    "captions_train2014.json",
    "captions_val2014.json",
)


COCO_DOUBLE_WORDS = (
    "motor bike",
    "motor cycle",
    "air plane",
    "traffic light",
    "street light",
    "traffic signal",
    "stop light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "suit case",
    "sports ball",
    "baseball bat",
    "baseball glove",
    "tennis racket",
    "wine glass",
    "hot dog",
    "cell phone",
    "mobile phone",
    "teddy bear",
    "hair drier",
    "potted plant",
    "bow tie",
    "laptop computer",
    "stove top oven",
    "hot dog",
    "teddy bear",
    "home plate",
    "train track",
)
ANIMAL_WORDS = (
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "animal",
    "cub",
)
VEHICLE_WORDS = ("jet", "train")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_provenance(annotation_path: Path) -> dict[str, str]:
    missing = [name for name in ANNOTATION_FILES if not (annotation_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Official CHAIR requires all four COCO 2014 annotation files; missing: "
            + ", ".join(missing)
        )
    return {name: sha256_file(annotation_path / name) for name in ANNOTATION_FILES}


def official_source_provenance(chair_root: Path) -> dict[str, str]:
    chair_source = chair_root / "utils" / "chair.py"
    synonyms = chair_root / "data" / "synonyms.txt"
    if not chair_source.is_file() or not synonyms.is_file():
        raise FileNotFoundError(
            f"{chair_root} is not an official LisaAnne/Hallucination checkout"
        )
    chair_sha = sha256_file(chair_source)
    synonyms_sha = sha256_file(synonyms)
    if chair_sha != OFFICIAL_CHAIR_SHA256:
        raise ValueError(
            f"Official chair.py SHA256 is {chair_sha}, expected {OFFICIAL_CHAIR_SHA256}"
        )
    if synonyms_sha != OFFICIAL_SYNONYMS_SHA256:
        raise ValueError(
            f"Official synonyms.txt SHA256 is {synonyms_sha}, expected {OFFICIAL_SYNONYMS_SHA256}"
        )
    return {"chair.py": chair_sha, "synonyms.txt": synonyms_sha}


class OfficialChairEvaluator:
    """Compute CHAIR-S/I with the official repository's lexical rules."""

    def __init__(self, synonyms_path: Path):
        synonyms = [line.strip().split(", ") for line in synonyms_path.read_text().splitlines()]
        self.mscoco_objects: list[str] = []
        self.inverse_synonym_dict: dict[str, str] = {}
        for synonym_group in synonyms:
            self.mscoco_objects.extend(synonym_group)
            for synonym in synonym_group:
                self.inverse_synonym_dict[synonym] = synonym_group[0]
        self.mscoco_object_set = set(self.mscoco_objects)

        self.double_word_dict = {word: word for word in COCO_DOUBLE_WORDS}
        for animal in ANIMAL_WORDS:
            self.double_word_dict[f"baby {animal}"] = animal
            self.double_word_dict[f"adult {animal}"] = animal
        for vehicle in VEHICLE_WORDS:
            self.double_word_dict[f"passenger {vehicle}"] = vehicle
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"

    def caption_to_words(self, caption: str) -> tuple[list[str], list[str], list[int], list[str]]:
        words = [singularize(word) for word in nltk.word_tokenize(caption.lower())]

        index = 0
        collapsed_words: list[str] = []
        original_indices: list[int] = []
        while index < len(words):
            original_indices.append(index)
            double_word = " ".join(words[index : index + 2])
            if double_word in self.double_word_dict:
                collapsed_words.append(self.double_word_dict[double_word])
                index += 2
            else:
                collapsed_words.append(words[index])
                index += 1
        words = collapsed_words

        if "toilet" in words and "seat" in words:
            words = [word for word in words if word != "seat"]

        matched_indices = [
            original_indices[idx]
            for idx, word in enumerate(words)
            if word in self.mscoco_object_set
        ]
        matched_words = [word for word in words if word in self.mscoco_object_set]
        node_words = [self.inverse_synonym_dict[word] for word in matched_words]
        return matched_words, node_words, matched_indices, collapsed_words

    def build_ground_truth(
        self,
        annotation_path: Path,
        image_ids: Iterable[int],
    ) -> dict[int, set[str]]:
        requested = {int(image_id) for image_id in image_ids}
        image_objects: dict[int, set[str]] = {image_id: set() for image_id in requested}

        for split in ("train", "val"):
            with (annotation_path / f"instances_{split}2014.json").open(
                "r", encoding="utf-8"
            ) as handle:
                payload = json.load(handle)
            id_to_name = {int(cat["id"]): str(cat["name"]) for cat in payload["categories"]}
            for annotation in payload["annotations"]:
                image_id = int(annotation["image_id"])
                if image_id in requested:
                    category = id_to_name[int(annotation["category_id"])]
                    image_objects[image_id].add(self.inverse_synonym_dict[category])

        for split in ("train", "val"):
            with (annotation_path / f"captions_{split}2014.json").open(
                "r", encoding="utf-8"
            ) as handle:
                payload = json.load(handle)
            for annotation in payload["annotations"]:
                image_id = int(annotation["image_id"])
                if image_id in requested:
                    _, node_words, _, _ = self.caption_to_words(str(annotation["caption"]))
                    image_objects[image_id].update(node_words)

        return image_objects

    def compute(
        self,
        captions: list[dict[str, Any]],
        image_objects: dict[int, set[str]],
    ) -> dict[str, Any]:
        sentences: list[dict[str, Any]] = []
        hallucinating_captions = 0
        hallucinated_mentions = 0
        object_mentions = 0
        recalled_objects = 0
        ground_truth_objects = 0

        for caption_row in captions:
            image_id = int(caption_row["image_id"])
            caption = str(caption_row["caption"])
            if image_id not in image_objects:
                raise ValueError(f"No COCO annotations were loaded for image_id={image_id}")
            words, node_words, indices, raw_words = self.caption_to_words(caption)
            ground_truth = image_objects[image_id]
            hallucinated = [
                (word, node_word, index)
                for word, node_word, index in zip(words, node_words, indices)
                if node_word not in ground_truth
            ]
            unique_recalled = set(node_words).intersection(ground_truth)
            sentence_chair_s = int(bool(hallucinated))
            sentence_chair_i = len(hallucinated) / len(words) if words else 0.0
            sentence_recall = len(unique_recalled) / len(ground_truth) if ground_truth else 0.0

            hallucinating_captions += sentence_chair_s
            hallucinated_mentions += len(hallucinated)
            object_mentions += len(node_words)
            recalled_objects += len(unique_recalled)
            ground_truth_objects += len(ground_truth)
            sentences.append(
                {
                    "image_id": image_id,
                    "caption": caption,
                    "mscoco_hallucinated_words": [list(item[:2]) for item in hallucinated],
                    "mscoco_gt_words": sorted(ground_truth),
                    "mscoco_generated_words": node_words,
                    "hallucination_idxs": [int(item[2]) for item in hallucinated],
                    "words": raw_words,
                    "metrics": {
                        "CHAIRs": sentence_chair_s,
                        "CHAIRi": sentence_chair_i,
                    },
                    "auxiliary_metrics": {"object_recall": sentence_recall},
                }
            )

        count = len(captions)
        if count == 0:
            raise ValueError("Cannot score an empty caption set")
        chair_s = hallucinating_captions / count
        chair_i = hallucinated_mentions / object_mentions if object_mentions else 0.0
        recall = recalled_objects / ground_truth_objects if ground_truth_objects else 0.0
        return {
            "sentences": sentences,
            "overall_metrics": {"CHAIRs": chair_s, "CHAIRi": chair_i},
            "auxiliary_metrics": {"object_recall": recall},
            "counts": {
                "captions": count,
                "hallucinating_captions": hallucinating_captions,
                "hallucinated_object_mentions": hallucinated_mentions,
                "generated_object_mentions": object_mentions,
                "recalled_ground_truth_objects": recalled_objects,
                "ground_truth_objects": ground_truth_objects,
            },
        }
