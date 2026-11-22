"""Select complete DeepFashion2 images using split-wide garment instance counts."""

import csv
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from PIL import Image
from shapely.errors import GEOSException

from convert_annotations import CLASSES, load_labels, publish_dataset, read_instances, save_labels


VAL_SIZE = 500
VAL_SEED = 174
VAL_MANIFEST = Path(__file__).parent / "splits" / "deepfashion2-val500.txt"
MIN_SHARE_RATIO = 0.80


def census(archive, source, skipped):
    """Count identifiable classes even when another item makes an image ineligible."""
    members = [i for i in archive.infolist() if not i.is_dir()]
    images = {Path(i.filename).stem for i in members
              if i.filename.startswith(f"{source}/image/") and i.filename.endswith(".jpg")}
    annotations = [i for i in members
                   if i.filename.startswith(f"{source}/annos/") and i.filename.endswith(".json")]
    seen, eligible, full = set(), {}, np.zeros(len(CLASSES), dtype=np.int64)
    for i, info in enumerate(annotations, 1):
        stem = Path(info.filename).stem
        if stem in seen:
            raise ValueError(f"duplicate annotation: {info.filename}")
        seen.add(stem)
        items = []
        try:
            obj = json.loads(archive.read(info))
            if not isinstance(obj, dict):
                raise ValueError("expected a per-image JSON object")
            items = [v for k, v in obj.items() if k.startswith("item") and k[4:].isdigit()]
            ids = []
            for item in items:
                category = item.get("category_id") if isinstance(item, dict) else None
                if type(category) is int and 1 <= category <= len(CLASSES):
                    ids.append(category - 1)
                    full[category - 1] += 1
            if stem not in images:
                raise ValueError("missing JPEG")
            read_instances(obj)
            eligible[stem] = ids
        except (ValueError, TypeError, KeyError) as error:
            skipped.append((source, stem, "census", len(items), str(error)))
        if i % 50000 == 0:
            print(f"{source}: scanned {i:,}/{len(annotations):,} JSONs", flush=True)
    for stem in sorted(images - seen):
        skipped.append((source, stem, "census", 0, "missing JSON"))
    stems = sorted(eligible)
    counts = np.zeros((len(stems), len(CLASSES)), dtype=np.int32)
    for row, stem in enumerate(stems):
        for cls in eligible[stem]:
            counts[row, cls] += 1
    return stems, counts, full, len(images), len(annotations)


def select_images(counts, target, size, seed):
    """Start at a seeded random sample; swap whole images to reduce class count error."""
    if not 1 <= size <= len(counts):
        raise ValueError(f"requested {size} images from {len(counts)} eligible images")
    rng = np.random.Generator(np.random.PCG64(seed))
    permutation = rng.permutation(len(counts))
    chosen, reserve = permutation[:size], permutation[size:]
    total = counts[chosen].sum(axis=0)
    weights = 1 / np.maximum(target, 1)
    for slot in rng.permutation(size):
        if not len(reserve):
            break
        trials = rng.choice(len(reserve), min(64, len(reserve)), replace=False)
        delta = counts[reserve[trials]] - counts[chosen[slot]]
        # Change in sum((observed - target)**2 / max(target, 1)).
        changes = ((2 * (total - target) + delta) * delta) @ weights
        best = changes.argmin()
        if changes[best] < -1e-10:
            j = trials[best]
            chosen[slot], reserve[j] = reserve[j], chosen[slot]
            total += delta[best]
    return chosen, reserve, total


def fixed_validation(stems):
    """Resolve the exact reviewed validation IDs in the supplied validation archive."""
    lines = VAL_MANIFEST.read_text(encoding="utf-8").splitlines()
    if len(lines) != VAL_SIZE or len(set(lines)) != VAL_SIZE:
        raise ValueError(f"expected {VAL_SIZE} unique images in {VAL_MANIFEST}")
    positions = {f"./images/val/{stem}.jpg": i for i, stem in enumerate(stems)}
    for name in lines:
        if name not in positions:
            raise ValueError(f"fixed validation image is missing or ineligible: {name}")
    return np.array([positions[name] for name in lines])


def convert_image(archive, source, stem, stage, split, expected):
    obj = json.loads(archive.read(f"{source}/annos/{stem}.json"))
    items = read_instances(obj)
    jpeg = archive.read(f"{source}/image/{stem}.jpg")
    # Validate the image in memory before publishing either member of the pair.
    try:
        with Image.open(BytesIO(jpeg)) as im:
            im.load()
            width, height = im.size
    except OSError as error:
        raise ValueError(f"unreadable JPEG: {error}") from error
    label = stage / "labels" / split / f"{stem}.txt"
    save_labels(label, items, width, height)
    saved = load_labels(label, width, height)
    observed = np.bincount([cls for cls, _ in saved], minlength=len(CLASSES))
    if not np.array_equal(observed, expected):
        raise RuntimeError(f"instance counts changed in {source}/{stem}")
    (stage / "images" / split / f"{stem}.jpg").write_bytes(jpeg)


def build_split(stage, zip_dir, source, split, size, seed, password, skipped):
    for folder in ("images", "labels"):
        (stage / folder / split).mkdir(parents=True)
    print(f"Counting garment instances in {source}.zip", flush=True)
    with ZipFile(zip_dir / f"{source}.zip") as archive:
        archive.setpassword(password)
        stems, counts, full, n_images, n_json = census(archive, source, skipped)
        if not n_images:
            raise ValueError(f"{source}.zip has no images")
        target = full * (size / n_images)
        if split == "val":
            chosen = fixed_validation(stems)
            total = counts[chosen].sum(axis=0)
            reserve = np.empty(0, dtype=int)
        else:
            chosen, reserve, total = select_images(counts, target, size, seed)
        print(f"{source}: {n_images:,} images, {full.sum():,} classified garments; "
              f"converting {size:,} images", flush=True)
        for slot in range(size):
            while True:
                index = chosen[slot]
                stem = stems[index]
                try:
                    convert_image(archive, source, stem, stage, split, counts[index])
                    break
                except (ValueError, GEOSException) as error:
                    if split == "val":
                        raise ValueError(f"cannot convert fixed validation image {stem}: {error}") from error
                    skipped.append((source, stem, "conversion", int(counts[index].sum()), str(error)))
                    (stage / "labels" / split / f"{stem}.txt").unlink(missing_ok=True)
                    if not len(reserve):
                        raise ValueError(f"insufficient complete images in {source}") from error
                    total -= counts[index]
                    scores = ((total + counts[reserve] - target) ** 2) @ (1 / np.maximum(target, 1))
                    j = scores.argmin()
                    chosen[slot] = reserve[j]
                    total += counts[reserve[j]]
                    reserve[j] = reserve[-1]
                    reserve = reserve[:-1]
            if (slot + 1) % 500 == 0:
                print(f"{split}: wrote {slot + 1:,}/{size:,} image-label pairs", flush=True)
    listing = stage / f"{split}.txt"
    listing.write_text("".join(f"./images/{split}/{s}.jpg\n" for s in sorted(stems[i] for i in chosen)),
                       encoding="utf-8")
    return full, total, {
        "source_images": n_images, "source_annotation_files": n_json,
        "source_classified_instances": int(full.sum()), "seed": seed,
        "images": size, "instances": int(total.sum()),
        "manifest_sha256": hashlib.sha256(listing.read_bytes()).hexdigest(),
    }


def prepare_subset(zip_dir, output, size, seed, password):
    if size < 1 or seed < 0:
        raise ValueError("train size must be positive and seed nonnegative")
    output.parent.mkdir(parents=True, exist_ok=True)
    skipped = []
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        full, train, train_info = build_split(stage, zip_dir, "train", "train", size, seed, password, skipped)
        _, val, val_info = build_split(stage, zip_dir, "validation", "val", VAL_SIZE, VAL_SEED, password, skipped)
        with (stage / "class_counts.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["yolo_id", "class", "full_train_instances", "full_train_pct",
                             "subset_instances", "subset_pct", "share_ratio", "flag", "val_instances"])
            for i, name in enumerate(CLASSES):
                p, q = full[i] / full.sum(), train[i] / train.sum()
                ratio = q / p if p else None
                flag = ""
                if p and not train[i]:
                    flag = "missing"
                elif p and ratio < MIN_SHARE_RATIO:
                    flag = "underrepresented"
                writer.writerow([i, name, full[i], f"{100*p:.6f}", train[i], f"{100*q:.6f}",
                                 f"{ratio:.6f}" if p else "", flag, val[i]])
                if flag:
                    print(f"{flag}: {name} ({100*q:.3f}% of subset vs {100*p:.3f}% of full train)")
        with (stage / "skipped.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source_split", "image_id", "stage", "garments_in_excluded_image", "reason"])
            writer.writerows(sorted(skipped))
        summary = {"train": train_info, "val": val_info, "minimum_share_ratio": MIN_SHARE_RATIO,
                   "excluded_images_by_split": dict(Counter(row[0] for row in skipped)),
                   "garments_in_excluded_images": sum(row[3] for row in skipped)}
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (stage / "data.yaml").write_text("train: train.txt\nval: val.txt\n"
                                         f"names: {json.dumps(CLASSES)}\n", encoding="utf-8")
        publish_dataset(stage, output, cache_files=[Path("labels/train.cache"), Path("labels/val.cache")])
    print(f"Prepared {size:,} training images ({train.sum():,} garments) and {VAL_SIZE} "
          f"validation images ({val.sum():,} garments).\nDataset: {output / 'data.yaml'}", flush=True)
    print(f"Excluded {len(skipped)} source images containing {summary['garments_in_excluded_images']} "
          f"garments. See {output / 'skipped.csv'} and {output / 'class_counts.csv'}.")
