# MVTec AD 2 (RD++): Offline Eval + Online Benchmark Submission

This repository now supports both:

1. **Offline evaluation on `test_public`** (for fast local iteration), and  
2. **Submission file generation for `test_private` / `test_private_mixed`** (for official benchmarking on https://benchmark.mvtec.com/).

Yes, the public-split evaluation path is still kept, so you can modify the model and test locally before uploading.

---

## 1) Environment

Recommended:

```bash
conda run -n anomaly-detection-py310 python -V
```

Install project deps (inside your env):

```bash
pip install -r requirements.txt
pip install -r MVTecAD2_public_code_utils/requirements.txt
```

---

## 2) Expected Dataset Layout (MVTec AD 2)

Set `DATA_ROOT=/path/to/mvtec_ad_2`, where object folders exist:

```text
mvtec_ad_2/
  can/
    train/good/*.png
    validation/good/*.png
    test_public/good/*.png
    test_public/bad/*.png
    test_public/ground_truth/bad/*_mask.png
    test_private/*.png
    test_private_mixed/*.png
  fabric/
  fruit_jelly/
  rice/
  sheet_metal/
  vial/
  wallplugs/
  walnuts/
```

---

## 3) Offline Evaluation (Public Split)

Use this when developing/debugging changes locally.

### Train

```bash
conda run -n anomaly-detection-py310 python main.py \
  --data_path "$DATA_ROOT" \
  --save_folder ./RDpp_ckpt \
  --classes can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts
```

### Evaluate on `test_public`

```bash
conda run -n anomaly-detection-py310 python inference.py \
  --data_path "$DATA_ROOT" \
  --checkpoint_folder ./RDpp_ckpt \
  --classes can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts
```

This writes local CSV metrics from public GT and is useful for offline checks only.

---

## 4) Generate Official Submission Files (Private Splits)

Use the single script:

```bash
conda run -n anomaly-detection-py310 bash run_ad2_submission.sh "$DATA_ROOT" ./ad2_run
```

What this does:

1. Trains RD++ per object (unless `--skip_train` is used),
2. Computes threshold from `validation` normals (`mean + 3*std`),
3. Exports required files:
   - `anomaly_images/.../*.tiff` (`float16`, single-channel),
   - optional `anomaly_images_thresholded/.../*.png`,
4. Runs official local checker:
   `MVTecAD2_public_code_utils/check_and_prepare_data_for_upload.py`,
5. Creates compressed archive for upload.

### Useful overrides

```bash
PYTHON_BIN="conda run -n anomaly-detection-py310 python" \
IMAGE_SIZE=256 \
BATCH_SIZE=16 \
NUM_WORKERS=4 \
EPOCHS=200 \
CLASSES="can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts" \
EXTRA_ARGS="--no_thresholded" \
bash run_ad2_submission.sh "$DATA_ROOT" ./ad2_run
```

If you already have checkpoints:

```bash
EXTRA_ARGS="--skip_train --checkpoint_root /path/to/checkpoints" \
conda run -n anomaly-detection-py310 bash run_ad2_submission.sh "$DATA_ROOT" ./ad2_run
```

---

## 5) Upload to Official Benchmark Server

After the script completes, upload the produced archive to:

- https://benchmark.mvtec.com/

The official `TESTpriv / TESTpriv,mix` metrics are only returned by the server.

---

## 6) Reproducing Paper-Style Reporting

- For local development: use `test_public` via `main.py` / `inference.py`.
- For official comparable numbers: submit private-split anomaly maps to benchmark server.
- Exact paper values are **not guaranteed** unless training/eval protocol and hyperparameters exactly match the paper setup and random variation.
