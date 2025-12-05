from ultralytics import YOLO

# Vortrainiertes Kennzeichen-Modell laden
model = YOLO("../weights/yolov8_plate.pt")

model.train(
    data="../my_finetune_data/data.yaml",  # Pfad zu deiner Daten-YAML
    epochs=60,          # 40–80 ist realistisch, CPU braucht einfach lang
    imgsz=640,          # ggf. auf 512 oder 416 senken, wenn es zu langsam ist
    batch=4,            # für deinen Laptop: klein halten
    lr0=2e-4,           # kleine Lernrate für Fine-Tuning
    freeze=0,           # NICHT einfrieren = volles Fine-Tuning
    workers=0,          # stabiler auf Windows/älterer CPU
    device="cpu"        # explizit CPU
)
