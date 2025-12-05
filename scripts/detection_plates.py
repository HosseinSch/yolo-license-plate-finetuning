import os
import cv2
from ultralytics import YOLO

# ============================================================
# KONFIGURATION
# ============================================================

# True  -> feingetuntes Modell verwenden
# False -> originales vortrainiertes Kennzeichen-Modell
USE_FINETUNED = True

# Pfade zu den Modellen
FINETUNED_MODEL_PATH = "./runs/detect/train2/weights/best.pt"
BASE_MODEL_PATH = "../weights/yolov8_plate.pt"

# Pfad zum Eingabe-Bilderordner
INPUT_FOLDER = "../images"

# Ausgabeordner
OUTPUT_IMG_FOLDER = "../output/images"      # annotierte Bilder
OUTPUT_PLATE_FOLDER = "../output/plates"   # ausgeschnittene Kennzeichen

# Inferenz-Parameter (sollten zu deinem Training passen)
IMG_SIZE = 640               # gleiche Größe wie beim Training
CONF_THRESH = 0.25           # Mindest-Konfidenz
IOU_THRESH = 0.45            # NMS IoU-Threshold

# ============================================================
# SETUP
# ============================================================

if USE_FINETUNED:
    model_path = FINETUNED_MODEL_PATH
else:
    model_path = BASE_MODEL_PATH

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Modell-Datei nicht gefunden: {model_path}")

if not os.path.isdir(INPUT_FOLDER):
    raise FileNotFoundError(f"Input-Ordner existiert nicht: {INPUT_FOLDER}")

os.makedirs(OUTPUT_IMG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_PLATE_FOLDER, exist_ok=True)

print(f"Verwende Modell: {model_path}")
print(f"Lese Bilder aus: {INPUT_FOLDER}")
print(f"Speichere bearbeitete Bilder in: {OUTPUT_IMG_FOLDER}")
print(f"Speichere ausgeschnittene Kennzeichen in: {OUTPUT_PLATE_FOLDER}")

# Modell laden (nutzt automatisch CPU bei dir)
model = YOLO(model_path)

# Nur Bilddateien berücksichtigen
image_files = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    print("Warnung: Keine .jpg/.jpeg/.png-Bilder im Input-Ordner gefunden.")
else:
    print(f"{len(image_files)} Bild(er) gefunden, starte Detektion...")

# ============================================================
# DETEKTION
# ============================================================

for filename in image_files:
    image_path = os.path.join(INPUT_FOLDER, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warnung: Bild konnte nicht gelesen werden: {image_path}")
        continue

    # YOLO-Inferenz mit definierten Parametern
    results = model.predict(
        source=image,        # direktes NumPy-Array
        imgsz=IMG_SIZE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        classes=[0],  # nur Klasse 0 = license_plate
        verbose=False
    )

    plates_found = 0

    # results ist eine Liste von Result-Objekten (normalerweise Länge 1)
    for result in results:
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        for j, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            # Annahme: Klasse 0 = license_plate (laut deinem data.yaml)
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Bounding-Box ins Bild zeichnen
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Optional: Label mit Konfidenztext
            conf = float(box.conf[0])
            label = f"plate {conf:.2f}"
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

            # Kennzeichen ausschneiden
            plate_crop = image[y1:y2, x1:x2]
            if plate_crop.size == 0:
                # Falls Box außerhalb des Bildes liegt
                continue

            base_name, _ = os.path.splitext(filename)
            plate_filename = f"{base_name}_plate{plates_found}.png"
            plate_output_path = os.path.join(OUTPUT_PLATE_FOLDER, plate_filename)
            cv2.imwrite(plate_output_path, plate_crop)
            plates_found += 1

    # Annotiertes Bild speichern
    output_path = os.path.join(OUTPUT_IMG_FOLDER, filename)
    cv2.imwrite(output_path, image)
    print(f"{filename}: {plates_found} Kennzeichen gefunden und gespeichert.")

print("Fertig! Alle Bilder wurden verarbeitet.")
