"""Prepare a portable YOLO dataset from the reviewed pilot or the DeepFashion2 ZIPs."""

import argparse
import csv
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from ultralytics.data.augment import RandomFlip
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import segments2boxes

from convert_annotations import CLASSES, draw_panel, load_labels, publish_dataset, zip_password
from deepfashion_subset import prepare_subset


def save_flip_preview(image, labels, target):
    """Apply YOLO's training transform to the source pixels and garment contours."""
    polygons = [np.asarray(rings[0], dtype=np.float32) for _, rings in labels]
    boxes = segments2boxes(polygons)
    # Instances takes a rectangular array. Repeat the final vertex to keep the
    # original outlines on both panels, even when garments have different lengths.
    points = max(map(len, polygons))
    segments = np.stack([np.pad(p, ((0, points - len(p)), (0, 0)), mode="edge") for p in polygons])
    instances = Instances(boxes, segments, bbox_format="xywh", normalized=False)
    flip = RandomFlip(p=1.0, direction="horizontal")
    flipped = flip({"img": np.asarray(image), "instances": instances})
    flipped_labels = [(cls, [ring[:len(parts[0])]])
                      for (cls, parts), ring in zip(labels, flipped["instances"].segments)]

    before = draw_panel(image, labels, "Original training image")
    after = draw_panel(Image.fromarray(flipped["img"]), flipped_labels, "Horizontal flip")
    preview = Image.new("RGB", (before.width * 2 + 2, before.height), "#777777")
    preview.paste(before)
    preview.paste(after, (before.width + 2, 0))
    preview.save(target)


def prepare(pilot, output):
    counts = {"train": [], "val": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        for split in counts:
            for folder in ("images", "labels"):
                (stage / folder / split).mkdir(parents=True)
        preview = None
        with (pilot / "matches.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                split = row["split"]
                source = {"train": "train", "val": "validation"}[split]
                member = Path(row["image_member"])
                stem = member.stem
                if (member != Path(source, "image", f"{stem}.jpg") or
                        row["annotation_member"] != f"{source}/annos/{stem}.json"):
                    raise ValueError(f"image/annotation mismatch in {split}: {member}")
                jpg = Path("images", split, member.name)
                txt = Path("labels", split, f"{stem}.txt")
                if (stage / jpg).exists():
                    raise ValueError(f"duplicate manifest entry: {jpg}")
                with Image.open(pilot / jpg) as image:
                    if image.size != (int(row["width"]), int(row["height"])):
                        raise ValueError(f"manifest dimensions differ for {jpg}")
                    labels = load_labels(pilot / txt, *image.size)
                    if (len(labels) != int(row["garments"]) or not labels or
                            [cls for cls, _ in labels] != list(map(int, row["yolo_class_ids"].split()))):
                        raise ValueError(f"manifest garments or classes differ for {txt}")
                    if split == "train" and preview is None:
                        preview = Path("overlays", f"train-{stem}-horizontal-flip.png")
                        (stage / preview.parent).mkdir()
                        save_flip_preview(image.convert("RGB"), labels, stage / preview)
                shutil.copyfile(pilot / jpg, stage / jpg)
                shutil.copyfile(pilot / txt, stage / txt)
                counts[split].append(len(labels))
        if not all(counts.values()):
            raise ValueError("the pilot manifest must contain training and validation examples")
        # Ultralytics resolves both paths from the YAML's directory on Elena's machine.
        (stage / "data.yaml").write_text(
            "train: images/train\nval: images/val\n" + f"names: {json.dumps(CLASSES)}\n",
            encoding="utf-8",
        )
        caches = [Path("labels", f"{split}.cache") for split in counts]
        publish_dataset(stage, output, cache_files=caches)
    for split, garments in counts.items():
        print(f"{split}: {len(garments)} images, {sum(garments)} garments")
    print(f"Dataset: {output / 'data.yaml'}\nTraining flip: {output / preview}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--pilot", type=Path, help="repackage the reviewed pilot (default mode)")
    inputs.add_argument("--zip-dir", type=Path, help="select a training subset from the original ZIPs")
    parser.add_argument("--train-size", type=int, help="training images in ZIP mode (default 2000)")
    parser.add_argument("--seed", type=int, help="training selection seed in ZIP mode (default 42)")
    parser.add_argument("--password-file", type=Path, help="local text file with the issued ZIP password")
    parser.add_argument("--output", type=Path, help="new output directory, or an identical previous build")
    args = parser.parse_args()
    try:
        if args.zip_dir:
            size = args.train_size if args.train_size is not None else 2000
            seed = args.seed if args.seed is not None else 42
            output = args.output or Path(f"data/deepfashion2-subset-{size}-seed{seed}")
            prepare_subset(args.zip_dir, output.resolve(), size, seed, zip_password(args.password_file))
        else:
            if args.train_size is not None or args.seed is not None or args.password_file:
                parser.error("--train-size, --seed and --password-file require --zip-dir")
            prepare((args.pilot or Path("data/deepfashion2-pilot")).resolve(),
                    (args.output or Path("data/deepfashion2-prepared")).resolve())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
