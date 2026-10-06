# Few-Shot Prototype Head Adaptation for On-Device ECG Personalization on PSoC 6

Code, measured results, manuscript, and video graphical abstract for the revised IEEE *Embedded Systems Letters* manuscript by Guilherme Silva, Pedro Silva, Gladston Moreira, and Eduardo Luz. This is a revision package; journal acceptance and a DOI have not yet been assigned.

- [Revised four-page manuscript](paper/main.pdf)
- [Video graphical abstract](paper/video_abstract/video_abstract.mp4), and [overlay](paper/video_abstract/overlay.png)

## What was measured

A 1,314-parameter one-dimensional CNN is trained on the MIT-BIH Arrhythmia Database DS1 patients. Its 32-dimensional feature extractor is frozen. On a new patient, each labeled support beat contributes to one class mean; a query is assigned to its nearest class mean. The PSoC 6 firmware also implements a linear head updated with 50 full-batch SGD steps for comparison. The task in this paper is binary normal versus pooled abnormal classification.

The table below reports mean **per-patient macro-F1** over every query beat of each eligible held-out DS2 record. The 22-record, no-adaptation DS2 score is 0.664; the pre-adaptation values below differ because the eligible patient set changes with the support requirement.

| Support per class | Eligible / excluded DS2 records | Pre | Linear SGD | Prototype | 1-NN |
|:---:|:---:|---:|---:|---:|---:|
| 1 | 18 / 4 | 0.635 | 0.671 | **0.731** | 0.731 |
| 5 | 15 / 7 | 0.639 | 0.704 | **0.771** | 0.765 |
| 10 | 14 / 8 | 0.646 | 0.709 | 0.797 | 0.797 |

The 1-NN baseline is k-nearest neighbors with **k = 1**. For each query it compares all `K × C` stored support embeddings, then uses the label of the single closest instance. With two classes, that is 2, 10, or 20 candidate embeddings for `K = 1, 5, 10`; each embedding has 32 values. Prototype inference compares only two class means. At one shot, the two rules are identical.

### Physical PSoC 6 replay

The CY8CPROTO-063-BLE Cortex-M4F was tested with deterministic replay episodes and at most 32 query beats per eligible record. Host and device scores below use the **same capped episodes**. They should not be compared directly with the full-DS2 scores above.

| K | Episodes / queries | Prototype host / device F1 | Linear SGD host / device F1 | CM4 prototype / SGD adaptation |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 18 / 576 | 0.798114 / 0.798222 | 0.688759 / 0.688778 | 22.83 / 25.97 ms |
| 5 | 15 / 480 | 0.784914 / 0.784933 | 0.732297 / 0.732333 | 114.04 / 124.52 ms |
| 10 | 14 / 448 | 0.803623 / 0.803643 | 0.682448 / 0.682357 | 228.06 / 247.69 ms |

The largest absolute host/device mean prototype F1 difference is 0.0001083. DWT profiling at 100 MHz measured approximately 11.39 ms per already segmented beat; acquisition and R-peak detection were not profiled. Adaptation timing includes support embedding extraction and the head update; it excludes query inference, UART, and replay control. At K=10, prototypes reduce the implemented SGD calibration time by 19.63 ms (7.9%). The 5.2 KB flash / 22.2 KB SRAM figure is a **model-centric** budget, not the total firmware image: SRAM includes 19,264 bytes of static activation arrays, 2,560 bytes of cached support embeddings, a 128-byte query embedding, and 264 bytes of prototype/counter state. A streaming prototype-only path maintains 264 bytes of adaptive state in addition to backbone working memory.

The restricted normal-only mode updates only the normal class mean and inherits the abnormal mean from DS1. On the same eligible records and identical query beats as full adaptation within each K, its macro-F1 is 0.678 / 0.689 / 0.697 for `K = 1 / 5 / 10`, compared with 0.635 / 0.639 / 0.646 before calibration. The eligible cohort can differ across K. These results use reference-annotated, segmented beats; upstream R-peak detection is not included. The validated firmware uses native-C float32, not INT8/TFLM.

## Reproduce the Python experiments

Use Python 3.11 or newer, then install dependencies from `requirements.txt`:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

The MIT-BIH signal files and model checkpoint are not redistributed. Run the pipeline from this directory to download the data, prepare the fixed inter-patient split, train the backbone, evaluate all three shot counts, and export model/replay headers:

```bash
bash scripts/run_all_offline.sh
```

The included video graphical abstract is the narrated, 157-second ManimGL export corresponding to this revision.

The run may take longer than the original benchmark machine. Per-record reference outputs used for the revision are already preserved in [`results/offline/`](results/offline/) and [`results/revision_round1/`](results/revision_round1/). To rerun only an offline shot count after preparing data and training:

```bash
python scripts/evaluate_pre_adaptation.py --config configs/personalization/protocol_5shot.yaml
```

For the paired normal-only ablation plotted in the paper, run `python scripts/evaluate_paired_normal_only.py` after preparing data and the checkpoint; it reuses the full-protocol query sets. To audit the DS1-only SGD sensitivity grid, run `python scripts/run_sgd_validation_sensitivity.py`. The revision keeps the prespecified learning rate 0.05 and 50 full-batch updates; neither the tiny baseline nor the scaling table selects settings on DS2.
The archived `results/head_retune/` grid is exploratory and is **not** the selection rule used for Table III.

## Reproduce the board replay

The board project is generated locally with ModusToolbox 3.8, then connected to the firmware sources in this repository. See [the firmware instructions](firmware/psoc6/README.md) and [the board setup notes](docs/PSOC6-063-BLE.md). The checked-in `firmware/psoc6/generated/benchmark_replay.h` corresponds to the one-shot replay. To switch shots, first export the matching support set and then regenerate this header. For example, for five shots:

```bash
python scripts/export_replay_support.py --config configs/personalization/protocol_5shot.yaml --out_dir firmware/psoc6/generated/replay_support
python scripts/export_firmware_benchmark_data.py --manifest firmware/psoc6/generated/replay_support/manifest.json --out firmware/psoc6/generated/benchmark_replay.h --summary-out results/firmware/ondevice_batch_5shot.json
python scripts/evaluate_firmware_benchmark_reference.py --summary results/firmware/ondevice_batch_5shot.json --out results/firmware/ondevice_batch_5shot_reference.json
make -C firmware/psoc6 build
make -C firmware/psoc6 program
```

Programming changes the connected board. The [captured UART logs](results/firmware/) and [consolidated summary](results/revision_round1/summary.json) document the original physical runs. The repository also contains `tests/test_personalization.py`, including a full-batch SGD reference check and the one-shot 1-NN/prototype equivalence check.

## Package layout

| Path | Contents |
|---|---|
| `paper/` | Revised manuscript PDF, figure assets, and video graphical abstract |
| `src/`, `scripts/`, `configs/` | Training, adaptation, evaluation, export, and repeatable configurations |
| `firmware/psoc6/` | Native-C backbone and heads, replay application, exported headers, and ModusToolbox setup |
| `results/offline/` | Full eligible-DS2 per-record evaluations |
| `results/firmware/` | Replay manifests, host/device scores, raw UART captures, and DWT cycles |
| `ui/`, `docs/` | Optional ECG demonstration and hardware documentation |

The full firmware image includes board support and flash-resident replay data. The published model-centric memory budget is not a measurement of the complete image. For methodological limitations and the scope of comparison with richer prototype estimators, see the [manuscript](paper/main.pdf).

## Dataset, citation, and license

MIT-BIH data are obtained separately from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) and remain subject to PhysioNet's terms. The split follows de Chazal *et al.* (IEEE T-BME, 2004). This code is licensed under [MIT](LICENSE). Please cite the manuscript title and authors above; replace the revision status with the final DOI when one is assigned.
