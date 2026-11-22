# DeepFashion2 preprocessing

For the training subset and Elena’s fixed validation set, follow [Reproduce the training subset](#reproduce-the-training-subset). The [training handoff](../training/DATA_HANDOFF.md) includes the trainer configuration and augmentation check. For the original 20/10 visual review, use the pilot workflow below.

The pilot converts 20 training images and 10 validation images into a YOLO segmentation dataset. It also saves a side-by-side polygon comparison for each image. Run the commands below from the repository root on `feature/data-pipeline`.

## Set up the environment

The reviewed build used Python 3.11.5, Pillow 10.4.0, and Shapely 2.2.0. To reproduce it in a virtual environment on macOS or Linux, run

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt 'Pillow==10.4.0' 'Shapely==2.2.0'
```

Place `train.zip` and `validation.zip` in the same directory and have the dataset-issued ZIP password ready. The converter reads the archives directly. Each split pairs `image/<id>.jpg` with `annos/<id>.json` and reads the polygons from each garment’s `segmentation` field. The separate `json_for_validation.zip` contains keypoint and retrieval evaluation data. The test ZIP contains images and evaluation metadata.

## Build or reproduce the pilot

```bash
python3 preprocessing/convert_annotations.py \
  --zip-dir "$HOME/Downloads/deepfashion_zipfiles" \
  --train-count 20 --val-count 10 \
  --output data/deepfashion2-pilot
```

Enter the password at the hidden prompt. If your archives are elsewhere, change `--zip-dir`. For a noninteractive build, the converter also accepts `--password-file /path/to/local/password-file`.

The converter selects images in filename order. If the first images in a split all have one garment, it uses the last slot for the first later multi-garment image it finds. Each garment produces one label row with a zero-based class ID followed by normalized `x y` coordinate pairs. The converter merges overlapping polygons within each garment, clips them to the image boundary, and joins any remaining rings into one contour.

The command prints the image and garment counts and a multi-garment overlay path for each split. You can rerun it against byte-identical output. To build a different sample, change the counts and choose a new `--output` directory.

## Review the build

The left panel draws the original JSON polygons. The right panel reads the saved YOLO label from disk and draws it on the same image. Compare each garment’s color, class, and filled area. Internal outlines can disappear where overlapping source components merge. The following overlay paths are relative to the generated `data/deepfashion2-pilot/` folder.

| Split | Expected images | Expected garments | Image and annotation inside the ZIP | Multi-garment overlay |
| --- | ---: | ---: | --- | --- |
| Train | 20 | 30 | `train/image/000001.jpg` and `train/annos/000001.json` | `overlays/train/000001.png` (short sleeve top and trousers) |
| Validation | 10 | 13 | `validation/image/000001.jpg` and `validation/annos/000001.json` | `overlays/val/000001.png` (vest and sling dress) |

The reviewed pilot’s lowest per-garment mask intersection-over-union was 0.9989 on training image `000018`, visible in `overlays/train/000018.png`. All 13 validation garments scored 1.0000.

Under `data/deepfashion2-pilot/`, `images/{train,val}/` holds the original JPEGs, `labels/{train,val}/` holds their matching YOLO text files, and `overlays/{train,val}/` holds the comparison PNGs. The generated `matches.csv` records both source paths, image size, garment count, class IDs, and component counts. The pilot’s `data.yaml` records the absolute path to the output directory on the build machine.

## Prepare the reviewed sample for Elena’s notebook

With the environment above active, run this command from the repository root. It uses the pilot manifest and the converted labels, so the ZIP password is only needed for the earlier conversion step.

```bash
python3 preprocessing/prepare_dataset.py \
  --pilot data/deepfashion2-pilot \
  --output data/deepfashion2-prepared
```

The prepared folder contains the same 20 training images and 30 garments, and 10 validation images and 13 garments. Each JPEG and label stays in its supplied split. The generated `data/deepfashion2-prepared/data.yaml` resolves both image directories relative to its own location, so Elena can copy the whole folder to another machine. Rerunning the command checks the generated files byte-for-byte.

To inspect this smaller pilot, set `DEEPFASHION2_DATA=data/deepfashion2-prepared/data.yaml` before running the first cell of `training/train.ipynb`. Start the kernel in the repository root or `training/`. It uses Ultralytics 8.4.215 to load four images per split and check that each garment has a nonempty mask. The validation loader runs with augmentation disabled. [Executable notebook cell](../training/train.ipynb)

The pilot preview applies Ultralytics’ horizontal flip to the training image and both of its garment contours. The orange trousers move from the lower right to the lower left. The generated overlay is `data/deepfashion2-prepared/overlays/train-000001-horizontal-flip.png`.

## Reproduce the training subset

Run from the repository root with Python 3.11. The reference build used NumPy 2.4.6, Pillow 10.4.0, Shapely 2.2.0, and Ultralytics 8.4.215. Activate the virtual environment described above, then run

```bash
python3 -m pip install -r requirements.txt 'numpy==2.4.6' 'Pillow==10.4.0' 'Shapely==2.2.0'
python3 preprocessing/prepare_dataset.py \
  --zip-dir "$HOME/Downloads/deepfashion_zipfiles" \
  --train-size 2000 --seed 42 \
  --output data/deepfashion2-subset-2000-seed42
```

Enter the issued ZIP password at the prompt. For an unattended run, append `--password-file /path/to/local/password-file`. The inputs are `train.zip` and `validation.zip`. The script streams their JSON files and writes only selected JPEGs and labels. The test archive stays held out. For another experiment, change `--train-size` or `--seed` and choose a fresh output directory. Running the same command again rebuilds the subset and checks every generated file byte-for-byte, allowing for the label caches Ultralytics creates.

### How selection works

1. Count every garment with a recognized category in every training JSON, including instances in images later excluded for invalid geometry. An image is eligible when it has a matching JPEG and all of its garment annotations pass the pilot converter’s checks.
2. For each class, target its full training instance count multiplied by the requested image count divided by the full training image count. Start with a NumPy PCG64 permutation of the eligible filenames in sorted order. In a seeded pass over the sample, try up to 64 randomly chosen replacements per slot. Accept the best swap only if it reduces the sum of squared count errors divided by each class target (with a minimum denominator of one). Each swap moves a complete image, including repeated garments of the same class.
3. Convert all selected items with the pilot’s polygon converter and 13-class mapping. If a training image fails conversion, log the whole image and try a replacement that best restores the class counts. The output percentages use garment instances as their denominator. `class_counts.csv` flags a class as `missing` at zero selected instances, or `underrepresented` when its subset share is below 80% of its full-training share.
4. Validation uses 500 filenames frozen from the original validation archive. They were initially selected with the same procedure against validation’s own class counts and seed 174. Subsequent builds read those names directly, independently of training size and seed. A missing or unusable fixed validation example stops the build. [Fixed validation filenames](splits/deepfashion2-val500.txt)

### Check the outputs and run Elena’s loader

The new folder contains `images/{train,val}/`, `labels/{train,val}/`, and a portable `data.yaml` pointing to the sorted `train.txt` and `val.txt` image lists. `class_counts.csv` records class counts, percentages, share ratios, and flags. `skipped.csv` records excluded image IDs, affected garment counts, and reasons. `summary.json` records split sizes, seeds and SHA-256 hashes of the two lists. [Dataset section and counts](../data/README.md)

To compare the manifests with the recorded hashes and confirm that validation matches the frozen list, run

```bash
shasum -a 256 data/deepfashion2-subset-2000-seed42/{train,val}.txt
cmp preprocessing/splits/deepfashion2-val500.txt data/deepfashion2-subset-2000-seed42/val.txt
```

The subset YAML is the notebook default. To execute both the manifest check and the segmentation trainer’s batch check from a terminal, run

```bash
python3 -m pip install ipython
DEEPFASHION2_DATA=data/deepfashion2-subset-2000-seed42/data.yaml \
  python3 -m IPython -c '%run training/train.ipynb'
```

The first cell audits the manifests, garment classes and nonempty masks. The second loads batches through the segmentation trainer and saves a before-and-after example at `data/review/trainer-horizontal-flip.png`. For the exact training call and local verification results, see [Elena’s handoff](../training/DATA_HANDOFF.md).
