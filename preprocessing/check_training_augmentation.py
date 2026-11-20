"""Check the segmentation trainer's returned masks against its pre-flip polygons."""

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from ultralytics import YOLO
from ultralytics.data.augment import Format, RandomFlip


REPO = Path(__file__).resolve().parents[1]
SUBSET = REPO / "data/deepfashion2-subset-2000-seed42/data.yaml"
PREVIEW = REPO / "data/review/trainer-horizontal-flip.png"
CONFIG = REPO / "configs/deepfashion2-train.yaml"


def panel(img, id_mask, classes, names, title):
    rgb = Image.fromarray(img.permute(1, 2, 0).contiguous().numpy())
    ids = np.array(Image.fromarray(id_mask.numpy().astype("int32")).resize(rgb.size, Image.Resampling.NEAREST))
    entries = sorted(set(classes))
    canvas = Image.new("RGB", (rgb.width, rgb.height + 34 + 18 * len(entries)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill="black")
    palette = ("#0084cf", "#dd5a19", "#12845b", "#a0419a", "#b39500", "#cc426c")
    for k, cls in enumerate(entries):
        color = palette[k % len(palette)]
        instance_ids = [j + 1 for j, c in enumerate(classes) if c == cls]
        mask = Image.fromarray((np.isin(ids, instance_ids) * 90).astype("uint8"))
        rgb.paste(color, mask=mask)
        y = rgb.height + 32 + 18 * k
        draw.rectangle((8, y, 19, y + 10), fill=color)
        draw.text((25, y - 2), names[cls], fill="black")
    canvas.paste(rgb, (0, 26))
    return canvas


def check_training_batches(data_yaml=SUBSET, preview=PREVIEW):
    """Return real CPU trainer batches and write one naturally flipped comparison."""
    settings = yaml.safe_load(CONFIG.read_text())
    # The data path only needs the architecture's stride. Weights are used by Elena's training run.
    yolo = YOLO("yolo26n-seg.yaml")
    with TemporaryDirectory() as tmp:
        trainer = yolo.task_map["segment"]["trainer"](overrides={
            **settings, "data": str(Path(data_yaml).resolve()), "model": "yolo26n-seg.yaml",
            "device": "cpu", "workers": 0, "project": tmp, "name": "batch-check",
        })
        trainer.model = yolo.model
        train = trainer.get_dataloader(trainer.data["train"], batch_size=settings["batch"], rank=-1, mode="train")
        val = trainer.get_dataloader(trainer.data["val"], batch_size=settings["batch"], rank=-1, mode="val")
        assert train.dataset.augment and not val.dataset.augment
        steps = train.dataset.transforms
        index, flip = next((i, t) for i, t in enumerate(steps.tolist())
                           if isinstance(t, RandomFlip) and t.direction == "horizontal")
        formatter = steps[index + 1]
        assert flip.p == settings["fliplr"] and isinstance(formatter, Format) and formatter.mask_overlap
        previous = []

        def capture(labels):
            previous.append(deepcopy(labels))
            return flip(labels)

        # Observe the real transform without changing its probability or the training sampler.
        steps[index] = capture
        try:
            for batch in train:
                example = None
                n_flipped = 0
                for i, before in enumerate(previous):
                    assert before["im_file"] == batch["im_file"][i]
                    left = formatter(deepcopy(before))
                    image = batch["img"][i]
                    mirrored = torch.equal(image, left["img"].flip(-1))
                    if mirrored:
                        n_flipped += 1
                        instances = before["instances"]
                        width = 1 if instances.normalized else image.shape[-1]
                        instances.segments[..., 0] = width - instances.segments[..., 0]
                    else:
                        assert torch.equal(image, left["img"])
                    # Mirror the reference vertices independently, then rasterize at the trainer's mask resolution.
                    expected = formatter(before)
                    classes = batch["cls"][batch["batch_idx"] == i].flatten()
                    assert torch.equal(expected["cls"].flatten(), classes)
                    assert torch.equal(expected["masks"][0], batch["masks"][i])
                    if mirrored and len(classes) and (example is None or len(classes) > example[0]):
                        example = (len(classes), i, left, classes.int().tolist())
                if example is not None:
                    break
                previous.clear()
            else:
                raise ValueError("no labeled horizontal flip observed in the training epoch")
        finally:
            steps[index] = flip
        _, i, left, classes = example
        names = trainer.data["names"]
        a = panel(left["img"], left["masks"][0], left["cls"].flatten().int().tolist(), names, "Before horizontal flip")
        b = panel(batch["img"][i], batch["masks"][i], classes, names, "After flip · returned training batch")
        overlay = Image.new("RGB", (a.width + b.width + 2, max(a.height, b.height)), "#dddddd")
        overlay.paste(a)
        overlay.paste(b, (a.width + 2, 0))
        preview = Path(preview)
        preview.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(preview)
        validation = next(iter(val))
        assert torch.any(validation["masks"]) and len(validation["cls"])
        print(f"Train: {len(batch['img'])} images, {len(batch['cls'])} instances, {n_flipped} flipped. "
              "All returned mask pixels matched the polygon reference.")
        print(f"Validation: {len(validation['img'])} images, {len(validation['cls'])} instances, "
              f"augment={val.dataset.augment}.")
        print(f"Preview: {preview} (batch index {i}, primary file {Path(batch['im_file'][i]).name})")
        return batch, validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=SUBSET)
    parser.add_argument("--preview", type=Path, default=PREVIEW)
    args = parser.parse_args()
    check_training_batches(args.data, args.preview)
