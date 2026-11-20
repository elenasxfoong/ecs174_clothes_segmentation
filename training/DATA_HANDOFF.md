# DeepFashion2 handoff for Elena

Use `data/deepfashion2-subset-2000-seed42/data.yaml` for the first training experiment. It describes 2,000 training images with 3,252 garments and 500 fixed validation images with 816 garments. The file is generated locally. Keep the prepared folder together when copying it to Colab or Drive. [Dataset section, class counts and limitations](../data/README.md)

## Prepare the data

1. Check out `feature/data-pipeline` and start in the repository root. Place the downloaded `train.zip`, `validation.zip`, `test.zip`, and `json_for_validation.zip` under `$HOME/Downloads/deepfashion_zipfiles/`. On another machine, use any directory and change `--zip-dir` below. The subset build reads only `train.zip` and `validation.zip` and prompts for the issued ZIP password.
2. In a Python 3.11 environment, install the dependencies and run the preparation command.

   ```bash
   python3 -m pip install -r requirements.txt 'numpy==2.4.6' 'Pillow==10.4.0' 'Shapely==2.2.0'
   python3 preprocessing/prepare_dataset.py \
     --zip-dir "$HOME/Downloads/deepfashion_zipfiles" \
     --train-size 2000 --seed 42 \
     --output data/deepfashion2-subset-2000-seed42
   ```

3. Confirm the printed counts above. The generated YAML uses the sibling `train.txt` and `val.txt` manifests. To confirm validation identity, run `cmp preprocessing/splits/deepfashion2-val500.txt data/deepfashion2-subset-2000-seed42/val.txt`. A successful comparison exits silently. [Preparation details and expected hashes](../preprocessing/README.md#reproduce-the-training-subset)

## Load a batch and inspect the augmentation

Run this cell with the kernel in the repository root or its `training/` directory. In Colab, first change the working directory to the checked-out repository. The cell is also saved as the second cell of the data-pipeline notebook. Set `DEEPFASHION2_DATA` to the YAML’s absolute path if the prepared folder is on Drive. [Notebook](train.ipynb)

```python
import os
import sys
from pathlib import Path

repo = Path.cwd().parent if Path.cwd().name == "training" else Path.cwd()
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
from preprocessing.check_training_augmentation import check_training_batches

data_yaml = repo / os.environ.get("DEEPFASHION2_DATA", "data/deepfashion2-subset-2000-seed42/data.yaml")
train_batch, val_batch = check_training_batches(data_yaml)
print(train_batch["img"].shape, val_batch["img"].shape)
```

Open `data/review/trainer-horizontal-flip.png`. The left panel shows the training image and masks immediately before the horizontal flip, after earlier augmentations such as mosaic and color adjustment. The right panel uses the image and mask tensors returned by the trainer. Running `python3 preprocessing/check_training_augmentation.py` from the repository root produces the same comparison with the default subset.

## Use the subset in the training notebook

The model-training notebook at commit `bd2534d` loads `yolo26n-seg.pt` and calls `model.train` on `coco8-seg.yaml`. Ultralytics 8.4.215 enables the training transform sequence through the dataset’s training mode. Its horizontal flip probability defaults to 0.5, and validation uses resizing and padding with augmentation disabled. The data-pipeline notebook’s first cell is a manifest smoke check with mosaic, scale and translation disabled. The cell above uses the segmentation trainer’s own loaders, including training mosaic and affine transforms. [Original training cell](https://github.com/elenasxfoong/ecs174_clothes_segmentation/blob/bd2534d3e48ae868f6d245e75a432828b767b8f9/training/train.ipynb)

After loading the batches, use the following call for the two-epoch GPU smoke test. The shared configuration specifies a batch size of four, image size 320 and horizontal flip probability 0.5. [Training configuration](../configs/deepfashion2-train.yaml)

```python
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")
results = model.train(
    cfg=str(repo / "configs/deepfashion2-train.yaml"),
    data=str(data_yaml),
    device=0,
    project="/content/ecs174_runs",
    name="deepfashion2_subset_smoke",
)
```

## Verification status

| Check | Locally verified on the prepared subset |
| --- | --- |
| Build and annotations | Two builds matched byte-for-byte. All 2,500 JPEG checksums matched their original archive members. Each image’s saved class sequence matched its source JSON, covering all 4,068 garments. |
| Trainer batch | On CPU, four 320 × 320 training images contained 20 garment instances after augmentation. One image flipped naturally. For all four, every returned mask pixel and class matched a reference rasterized from the pre-flip polygons with the observed horizontal transformation. |
| Visual comparison | Batch index 1, with primary source `033325.jpg`, shows a vest and sling dress. The lettering reverses and the colored masks follow the garments. The 80 × 80 training masks are enlarged for viewing. |
| Validation | The four-image batch contained five instances with augmentation disabled. The train and validation scans reported zero corrupt images. |

**Pending Elena’s confirmation.** The local review used the YOLO26n segmentation architecture and a CPU loader. On Colab, confirm the manifest sizes and hashes, open the regenerated overlay, then run the GPU cell with the pretrained weights. Record the resulting run directory and validation metrics with the experiment. These metrics become available after model training.
