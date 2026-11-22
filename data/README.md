# DeepFashion2 training subset

## Dataset and preprocessing

We used DeepFashion2 (Ge et al., 2019), a benchmark of consumer and commercial clothing photographs. In the downloaded release, each training or validation JPEG matches a same-stem JSON in its split’s `annos/` directory. Each garment has a category ID and one or more segmentation polygons. We converted these to one YOLO segment row per garment using zero-based class IDs and image-normalized coordinates. [Dataset and download instructions](https://github.com/switchablenorms/DeepFashion2) · [Dataset paper](https://arxiv.org/abs/1901.07973)

The original training split contained 312,186 garments in 191,961 images. With seed 42, we sampled 2,000 eligible images, then swapped whole images to bring each garment-class count closer to its full-training count scaled by 2,000/191,961. We kept all garments in each selected image, including multiple instances of the same class. The resulting training set has 3,252 garments in 2,000 images, of which 1,203 contain multiple garments.

We selected 500 images from the original validation split using its own class counts and seed 174, then froze their filenames for subsequent experiments. They contain 816 garments, with multiple garments in 289 images. Training and validation each cover all 13 categories. The supplied test split remains held out.

| Split | Downloaded images | Annotated garments in download | Images in this experiment | Garments in this experiment |
| --- | ---: | ---: | ---: | ---: |
| Training | 191,961 | 312,186 | 2,000 | 3,252 |
| Validation | 32,153 | 52,490 | 500 | 816 |
| Test | 62,629 | Segmentation ground truth unavailable | 0 | — |

## Class representation

Percentages are shares of all garment instances in the corresponding training set. Shorts had the largest absolute difference, about 0.0130 percentage points. All 13 classes exceeded the 80% relative-share threshold. Short sleeve outwear, the rarest class, has six training instances and two validation instances.

| Garment class | Full train | Full share | Subset | Subset share | Fixed validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| short sleeve top | 71,645 | 22.95% | 746 | 22.94% | 195 |
| long sleeve top | 36,064 | 11.55% | 376 | 11.56% | 93 |
| short sleeve outwear | 543 | 0.17% | 6 | 0.18% | 2 |
| long sleeve outwear | 13,457 | 4.31% | 140 | 4.31% | 31 |
| vest | 16,095 | 5.16% | 168 | 5.17% | 33 |
| sling | 1,985 | 0.64% | 21 | 0.65% | 5 |
| shorts | 36,616 | 11.73% | 381 | 11.72% | 65 |
| trousers | 55,387 | 17.74% | 577 | 17.74% | 149 |
| skirt | 30,835 | 9.88% | 321 | 9.87% | 101 |
| short sleeve dress | 17,211 | 5.51% | 179 | 5.50% | 49 |
| long sleeve dress | 7,907 | 2.53% | 82 | 2.52% | 23 |
| vest dress | 17,949 | 5.75% | 187 | 5.75% | 52 |
| sling dress | 6,492 | 2.08% | 68 | 2.09% | 18 |
| **Total** | **312,186** | **100%** | **3,252** | **100%** | **816** |

### Exclusions and limitations

Training images `047446`, `175722`, and `180057` each contained a polygon component with only two distinct vertices. Excluding these three whole images removed four garments from the eligible pool, including a valid skirt in `047446`. Their class IDs count toward the full-training baseline. Every annotated garment in the selected sets has a saved label row. Validation had zero exclusions. The build writes the individual reasons to `skipped.csv` and the full-precision percentages to `class_counts.csv`.

Matching class shares preserves the source imbalance. With just two short sleeve outwear instances in validation and five sling instances, a single error can strongly affect those classes’ scores. The sampling objective covers garment-class frequencies. Coverage by viewpoint, occlusion, and consumer versus shop source was unmeasured. For garments with disconnected visible regions, the converter joins the rings to fit YOLO’s one-contour-per-instance format, which can leave narrow connecting lines.

The test ZIP contains 62,629 JPEGs and three JSON files with keypoint and retrieval metadata. It supplies no garment segmentation polygons. Elena can use test images for inference and qualitative review of predicted masks. Local mask IoU and segmentation AP require ground truth. The fixed labeled validation set supports those metrics. The separate validation evaluation ZIP contains keypoint and retrieval data, while the garment contours for validation come from `validation.zip`.

## Reproduce and identify this build

The portable dataset YAML is generated at `data/deepfashion2-subset-2000-seed42/data.yaml`. Two local builds matched byte-for-byte. The four-image manifest smoke check loaded seven training instances and five validation instances. The separate trainer augmentation check verified the masks returned after mosaic, geometric and color transforms, and probabilistic horizontal flipping. [Preparation and loader commands](../preprocessing/README.md#reproduce-the-training-subset) · [Elena’s handoff and verification results](../training/DATA_HANDOFF.md)

| Image list | Images | SHA-256 of the list |
| --- | ---: | --- |
| Generated `train.txt` | 2,000 | `7e47a78c9ceb213b7e39e8d4d39c2b4de36171e122ba9e971cd90fc8089b39fd` |
| [Fixed validation](../preprocessing/splits/deepfashion2-val500.txt) | 500 | `ef6bb5401b83804811aeb8676dffd1f8ce9cfc19a4b2a20f8b0635d71b4b1591` |
