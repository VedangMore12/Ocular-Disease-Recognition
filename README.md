# Ocular Disease Recognition

## Run the Gradio app

From this directory, install the dependencies and start the local web app:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main_app.py
```

Open the local address printed in the terminal. The **Image analysis** tab accepts
one or both fundus images. The **Validation metrics** tab evaluates the loaded
checkpoint on the deterministic, patient-grouped validation split and displays
accuracy, balanced accuracy, per-class precision/recall/F1, and a confusion
matrix.

If `model/model_weights.pth` is missing or you want to retrain after changing
the training pipeline, run this from the project directory:

```powershell
python -m model.train_model
```

For a faster CPU iteration, use the smaller fine-tuning scope and fewer epochs:

```powershell
python -m model.train_model --fast --num-workers 2
```

`--fast` uses five epochs and updates only ResNet layer4 plus the classifier
head. It is intended for quick comparisons; it may not reach the same validation
performance as the full run. You can also set `--epochs`, `--batch-size`, and
`--num-workers` explicitly.

For an isolated faster experiment, use the cached trainer. It preprocesses
each image once and writes the cache, checkpoint, and reports only under
`model/cached_run/`, so it does not overwrite the existing
`model/model_weights.pth`:

```powershell
python -m model.train_cached --epochs 10 --batch-size 32 --num-workers 2
```

The cached trainer selects the best checkpoint by macro F1 by default. Use
`--selection-metric accuracy` if overall accuracy is the primary objective.
It automatically uses CUDA when a CUDA-enabled PyTorch installation and an
NVIDIA GPU are available. To require CUDA and fail clearly otherwise, use
`--device cuda`; `--device cpu` forces CPU training.

To recreate the isolated original-style baseline, run:

```powershell
python -m model.train_baseline --epochs 15 --batch-size 32 --num-workers 2
```

The baseline writes only to `model/baseline_run/` and does not overwrite any
other checkpoint. It uses standard resized/normalized images, a frozen
ResNet18 backbone, a trainable classification head, and unweighted
`BCEWithLogitsLoss`.

The Gradio app uses `model/model_weights.pth` by default. To switch weights
without copying or overwriting files, set `OCULAR_MODEL_PATH` before starting
the app:

```powershell
$env:OCULAR_MODEL_PATH = "C:\Users\vedan\Desktop\GIT\ocular_disease_tkinter\model\baseline_run\model_weights.pth"
python main_app.py
```

For the cached model:

```powershell
$env:OCULAR_MODEL_PATH = "C:\Users\vedan\Desktop\GIT\ocular_disease_tkinter\model\cached_run\model_weights.pth"
python main_app.py
```

Close the terminal or run `Remove-Item Env:OCULAR_MODEL_PATH` to return to the
default checkpoint.

Training and validation use filenames from `full_df.csv` and images under
`images/`. They do not read `test_images/`; keep that folder untouched until
the final test evaluation.

Training and validation now share the CLAHE/vessel preprocessing used by the
app. Training uses class-weighted cross-entropy, with weights calculated only
from the training patients, and fine-tunes ResNet layer3, layer4, and the head.
Retrain before interpreting validation results because an existing checkpoint
may have been trained with the previous preprocessing and loss. To continue
from an existing checkpoint instead of starting from ImageNet weights, run:

```powershell
python -m model.finetunemodel
```

This project is for educational use only; its predictions are not a medical
diagnosis.
