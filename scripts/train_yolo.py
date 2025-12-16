from ultralytics import YOLO

# Vortrainiertes Kennzeichen-Modell laden
model = YOLO("../weights/yolov8_plate.pt")

model.train(
    data="../my_finetune_data/data.yaml",  
    epochs=60,         
    imgsz=640,         
    batch=4,            
    lr0=2e-4,           
    freeze=0,           
    workers=0,          
    device="cpu"       
)
