"""Convert a deterministic DeepFashion2 pilot and compare its saved YOLO segments."""

import argparse
import csv
import filecmp
import getpass
import json
import math
import os
import sys
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

from PIL import Image, ImageDraw
from shapely import Polygon, box, make_valid, union_all


CLASSES = (
    "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear",
    "vest", "sling", "shorts", "trousers", "skirt", "short sleeve dress",
    "long sleeve dress", "vest dress", "sling dress",
)
COLORS = ("#0984e3", "#e17018", "#00865a", "#a549b8", "#db3965", "#796400")


def read_instances(annotation):
    """Return (zero-based class, component polygons) in garment number order."""
    items = sorted((int(k[4:]), v) for k, v in annotation.items()
                   if k.startswith("item") and k[4:].isdigit())
    if not items:
        raise ValueError("no garment items")
    instances = []
    for number, item in items:
        category = item["category_id"]
        if type(category) is not int or not 1 <= category <= len(CLASSES):
            raise ValueError(f"item{number} has invalid category {category!r}")
        components = []
        for coords in item["segmentation"]:
            if (len(coords) < 6 or len(coords) % 2 or
                    not all(isinstance(v, (int, float)) and math.isfinite(v) for v in coords)):
                raise ValueError(f"item{number} has invalid polygon coordinates")
            points = list(zip(coords[::2], coords[1::2]))
            if len(set(points)) < 3:
                raise ValueError(f"item{number} has a degenerate polygon")
            components.append(points)
        if not components:
            raise ValueError(f"item{number} has no segmentation polygons")
        instances.append((category - 1, components))
    return instances


def joined_contour(components):
    # YOLO allows one contour per instance. Visit each additional ring and return
    # along the same nearest-vertex bridge, retaining all its vertices.
    contour = list(components[0])
    for ring in components[1:]:
        _, i, j = min(((x - u)**2 + (y - v)**2, i, j)
                      for i, (x, y) in enumerate(contour)
                      for j, (u, v) in enumerate(ring))
        contour[i + 1:i + 1] = ring[j:] + ring[:j + 1] + [contour[i]]
    return contour


def polygon_parts(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geometry.geoms:
            yield from polygon_parts(part)


def save_labels(path, instances, width, height):
    lines = []
    frame = box(0, 0, width, height)
    for cls, components in instances:
        fields = [str(cls)]
        # Each source component is filled independently. Union them before
        # tracing, or shared sleeve/body regions can turn into holes in YOLO.
        polygons = [make_valid(Polygon(ring)) for ring in components]
        visible = union_all(polygons).intersection(frame)
        rings = [list(boundary.coords)[:-1]
                 for part in polygon_parts(visible)
                 for boundary in (part.exterior, *part.interiors)]
        if not rings:
            raise ValueError(f"{CLASSES[cls]} has no visible polygon")
        for x, y in joined_contour(rings):
            fields.extend((f"{x / width:.8f}", f"{y / height:.8f}"))
        lines.append(" ".join(fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_labels(path, width, height):
    instances = []
    for row in path.read_text(encoding="utf-8").splitlines():
        category, *tokens = row.split()
        cls = int(category)
        values = list(map(float, tokens))
        if (not 0 <= cls < len(CLASSES) or len(values) < 6 or len(values) % 2 or
                not all(0 <= value <= 1 for value in values)):
            raise ValueError(f"invalid YOLO segment in {path.name}")
        ring = [(x * width, y * height) for x, y in zip(values[::2], values[1::2])]
        instances.append((cls, [ring]))
    return instances


def draw_panel(image, instances, title):
    panel_width = max(image.width, 200)
    panel = Image.new("RGB", (panel_width, image.height + 34 + 20 * len(instances)), "white")
    painted = image.copy()
    legend = ImageDraw.Draw(panel)
    legend.text((8, 6), title, fill="black")
    for index, (cls, components) in enumerate(instances):
        color = COLORS[index % len(COLORS)]
        mask = Image.new("L", image.size)
        mask_draw = ImageDraw.Draw(mask)
        rings = [[(round(x), round(y)) for x, y in ring] for ring in components]
        for ring in rings:
            mask_draw.polygon(ring, fill=65)
        painted.paste(color, mask=mask)
        outlines = ImageDraw.Draw(painted)
        for ring in rings:
            outlines.line(ring + [ring[0]], fill=color, width=2)
        y = 33 + image.height + index * 20
        legend.rectangle((8, y, 17, y + 9), fill=color)
        legend.text((23, y - 2), f"{cls}  {CLASSES[cls]}", fill="black")
    panel.paste(painted, ((panel_width - image.width) // 2, 27))
    return panel


def sample_annotations(archive, source, image_ids, count):
    # Prefer the first IDs. If they are all single-garment, reserve the final
    # slot for the first later multi-garment example available in this split.
    selected = []
    for stem in sorted(image_ids):
        member = f"{source}/annos/{stem}.json"
        try:
            instances = read_instances(json.loads(archive.read(member)))
        except (ValueError, KeyError, RuntimeError) as error:
            raise ValueError(f"{member}: {error}") from error
        if len(selected) < count:
            selected.append((stem, member, instances))
        elif len(instances) > 1:
            selected[-1] = (stem, member, instances)
            break
        if len(selected) == count and any(len(items) > 1 for _, _, items in selected):
            break
    return selected


def convert_split(archive_path, split, count, root, password, manifest):
    source = "validation" if split == "val" else "train"
    for folder in ("images", "labels", "overlays"):
        (root / folder / split).mkdir(parents=True)
    total_items = 0
    with ZipFile(archive_path) as archive:
        archive.setpassword(password)
        names = archive.namelist()
        image_ids = {Path(n).stem for n in names if n.startswith(f"{source}/image/")
                     and n.endswith(".jpg")}
        anno_ids = {Path(n).stem for n in names if n.startswith(f"{source}/annos/")
                    and n.endswith(".json")}
        if image_ids != anno_ids:
            raise ValueError(f"{archive_path.name} has unmatched image/annotation IDs")
        if not 1 <= count <= len(image_ids):
            raise ValueError(f"sample count for {source} must be in [1, {len(image_ids)}]")
        sample = sample_annotations(archive, source, image_ids, count)
        for stem, anno_member, original in sample:
            image_member = f"{source}/image/{stem}.jpg"
            try:
                data = archive.read(image_member)
                (root / "images" / split / f"{stem}.jpg").write_bytes(data)
                with Image.open(BytesIO(data)) as jpeg:
                    image = jpeg.convert("RGB")
                label = root / "labels" / split / f"{stem}.txt"
                save_labels(label, original, *image.size)
                saved = load_labels(label, *image.size)
                if [c for c, _ in saved] != [c for c, _ in original]:
                    raise ValueError("saved classes or instance count differ from source")
                left = draw_panel(image, original, "DeepFashion2 annotation")
                right = draw_panel(image, saved, "Saved YOLO label")
                comparison = Image.new("RGB", (left.width * 2 + 2, left.height), "#777777")
                comparison.paste(left)
                comparison.paste(right, (left.width + 2, 0))
                comparison.save(root / "overlays" / split / f"{stem}.png")
                manifest.writerow([split, image_member, anno_member, image.width, image.height,
                                   len(original), " ".join(str(c) for c, _ in original),
                                   " ".join(str(len(rings)) for _, rings in original)])
                total_items += len(original)
            except (ValueError, KeyError, RuntimeError, OSError) as error:
                raise ValueError(f"{archive_path.name} / {stem}: {error}") from error
    multi = next((stem for stem, _, items in sample if len(items) > 1), None)
    return total_items, multi


def publish_dataset(root, output, *, cache_files=()):
    """Publish the staged dataset, or verify that an existing build is identical."""
    if not output.exists():
        root.rename(output)
        return
    files = {p.relative_to(root) for p in root.rglob("*") if p.is_file()}
    existing = {p.relative_to(output) for p in output.rglob("*") if p.is_file()}
    if (files != existing.difference(cache_files) or
            not all(filecmp.cmp(root / p, output / p, shallow=False) for p in files)):
        raise ValueError(f"{output} contains different files; choose a fresh --output")


def build(args, password):
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        root = Path(tmp) / "pilot"
        root.mkdir()
        totals = []
        with (root / "matches.csv").open("w", newline="", encoding="utf-8") as f:
            manifest = csv.writer(f)
            manifest.writerow(["split", "image_member", "annotation_member", "width", "height",
                               "garments", "yolo_class_ids", "components_per_garment"])
            for filename, split, count in (("train.zip", "train", args.train_count),
                                           ("validation.zip", "val", args.val_count)):
                total, multi = convert_split(args.zip_dir / filename, split, count, root, password, manifest)
                summary = f"{split}: {count} images, {total} garments"
                if multi:
                    summary += f" (multi-garment overlay: overlays/{split}/{multi}.png)"
                totals.append(summary)
        (root / "data.yaml").write_text(
            f"path: {json.dumps(str(output))}\ntrain: images/train\nval: images/val\n"
            f"names: {json.dumps(CLASSES)}\n", encoding="utf-8")
        publish_dataset(root, output)
    print("\n".join(totals))
    print(f"Overlays: {output / 'overlays'}")


def zip_password(password_file):
    if password_file:
        return password_file.read_bytes().rstrip(b"\r\n") or None
    password = os.environ.get("DEEPFASHION2_ZIP_PASSWORD", "")
    if not password and sys.stdin.isatty():
        password = getpass.getpass("DeepFashion2 ZIP password: ")
    return password.encode() or None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", type=Path, required=True, help="folder with train.zip and validation.zip")
    parser.add_argument("--output", type=Path, default=Path("data/deepfashion2-pilot"))
    parser.add_argument("--train-count", type=int, default=20)
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--password-file", type=Path, help="local text file with the issued ZIP password")
    args = parser.parse_args()
    try:
        build(args, zip_password(args.password_file))
    except (OSError, ValueError, RuntimeError, BadZipFile) as error:
        parser.exit(1, f"Conversion failed: {error}\n")


if __name__ == "__main__":
    main()
