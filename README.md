# YOLOv8 License Plate Detection – Fine-Tuning & Inference

This repository contains a complete workflow for **license plate detection** using **Ultralytics YOLOv8**:

- Fine-tuning a pre-trained license plate detector on your own dataset  
- Running inference on image folders  
- Automatically cropping detected license plates into separate images  

The project is optimized for **CPU-only training** (e.g. laptops without NVIDIA GPU).

---

## 1. Features

- ✅ Single-class model: detects **only** `license_plate`  
- ✅ Fine-tuning starting from a pre-trained YOLOv8 license plate model  
- ✅ CPU-friendly training setup (small batch size, low workers)  
- ✅ Batch detection: process all images in a folder  
- ✅ Automatic cropping and saving of each detected plate  

---

## 2. Project structure

Recommended directory layout:

```text
yolo_training/
├─ scripts/
│  ├─ train_yolo.py          # training script
│  └─ detection_plates.py    # inference + plate cropping
├─ weights/
│  └─ yolov8_plate.pt        # pre-trained plate model (from Kaggle)
├─ my_finetune_data/
   ├─ data.yaml              # dataset config
   ├─ images/
   │  ├─ train/
   │  └─ val/
   └─ labels/
      ├─ train/
      └─ val/
```

Paths in the scripts are set relative to this structure.

---

## 3. Requirements

- Python 3.9+  
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)  
- OpenCV

Install dependencies (ideally in a virtual environment):

```bash
pip install ultralytics opencv-python
```

Optional but recommended for clean environments:

```bash
pip install ipykernel
```

---

## 4. Dataset & `data.yaml`

This project uses a **single class** dataset: `license_plate`.

Typical dataset structure:

```text
my_finetune_data/
├─ data.yaml
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ train/
   └─ val/
```

Each image in `images/train` / `images/val` has a corresponding YOLO label file in `labels/train` / `labels/val`.

Example `data.yaml`:

```yaml
train: train/images
val: val/images
nc: 1
names: [
    0: 'license_plate'
]
```

Key points:

- `nc: 1` → model has exactly **one class**  
- `names[0] = 'license_plate'`  
- No other classes exist; the model will only predict license plates.

Place `data.yaml` inside `my_finetune_data/`.

---

## 5. Training (fine-tuning)

Training is done with `scripts/train_yolo.py`, starting from a **pre-trained license plate model** (e.g. downloaded from Kaggle into `weights/yolov8_plate.pt`).

Example `train_yolo.py`:

```python
from ultralytics import YOLO

# Load pre-trained license plate model
model = YOLO("../weights/yolov8_plate.pt")

# Fine-tune on your custom dataset
model.train(
    data="../my_finetune_data/data.yaml",  # dataset config
    epochs=60,                             # adjust depending on dataset size
    imgsz=640,                             # can be 512 if training is too slow
    batch=4,                               # small batch size for CPU
    lr0=2e-4,                              # suitable learning rate for fine-tuning
    freeze=0,                              # full fine-tuning (no layers frozen)
    workers=0,                             # stable on Windows / older CPUs
    device="cpu"                           # CPU-only training
)
```

Run training from the `scripts/` directory:

```bash
cd yolo_training/scripts
python train_yolo.py
```

The fine-tuned weights will be saved under something like:

```text
../runs/detect/train2/weights/best.pt
```

(Exact folder name may differ depending on Ultralytics version and existing runs.)

---

## 6. Inference & plate cropping

Inference and plate cropping are handled by `scripts/detection_plates.py`.

Key configuration at the top of the script:

```python
# True  -> use fine-tuned model
# False -> use original pre-trained model
USE_FINETUNED = True

FINETUNED_MODEL_PATH = "./runs/detect/train2/weights/best.pt"
BASE_MODEL_PATH      = "../weights/yolov8_plate.pt"

INPUT_FOLDER         = "../images"
OUTPUT_IMG_FOLDER    = "../output/images"
OUTPUT_PLATE_FOLDER  = "../output/plates"

IMG_SIZE    = 640
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
```

Core logic (simplified):

- Loads either:
  - your fine-tuned model (`best.pt`) or  
  - the base model  
- Reads all `.jpg`, `.jpeg`, `.png` images from `INPUT_FOLDER`
- Runs YOLOv8 inference with:
  - `imgsz=IMG_SIZE`
  - `conf=CONF_THRESH`
  - `iou=IOU_THRESH`
  - optionally restricted to class `0` (license_plate)
- Draws bounding boxes on the original image
- Saves:
  - annotated full images to `OUTPUT_IMG_FOLDER`
  - cropped license plates to `OUTPUT_PLATE_FOLDER`

Run inference from `scripts/`:

```bash
cd yolo_training/scripts
python detection_plates.py
```

Make sure:

- You put test images into `../images/`  
- Output folders `../output/images` and `../output/plates` exist or will be created automatically.

---

## 7. Only license plates (no other objects)

The system is designed so that **only license plates** are detected:

1. Dataset:  
   - `nc: 1`  
   - `names: ['license_plate']`  

2. Model head:  
   - Only one class output (class id `0`)

3. Inference filtering (inside `detection_plates.py`):

   ```python
   cls_id = int(box.cls[0])
   if cls_id != 0:
       continue
   ```

4. Optional: restrict prediction to class `0` directly in `model.predict`:

   ```python
   results = model.predict(
       source=image,
       imgsz=IMG_SIZE,
       conf=CONF_THRESH,
       iou=IOU_THRESH,
       classes=[0],  # only license_plate
       verbose=False
   )
   ```

With these settings, the model will not return any other classes.

---

## 8. Notes for CPU-only training

This setup is tailored for a typical laptop (e.g. Intel i5, no dedicated GPU):

- Use **small batch sizes** (`batch=2–4`)  
- Keep `workers=0` or `1` to avoid multiprocessing overhead on Windows  
- Consider reducing `imgsz` to `512` or `416` if training is too slow  
- Use modest `epochs` (e.g. `40–80`) and monitor validation metrics  

If training time is too long, you can:

- Lower `epochs`  
- Train in two stages (first fewer epochs, check metrics, then continue if needed)

---

## 9. Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework  
- Pre-trained license plate model obtained from a Kaggle dataset (yolov8 plate detection & fine-tuned weights)  

---

## 10. License

Specify your license here, for example:

```text
MIT License

Copyright (c) 2025 ...
```

Adapt this section according to your own licensing requirements.
