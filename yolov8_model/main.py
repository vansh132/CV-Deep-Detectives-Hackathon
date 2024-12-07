from ultralytics import YOLO

# # Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch

# # Use the model
# model.train(data="config.yaml", epochs=50, imgsz=640)  # train the model

# ----------------------------------------------------------------


from ultralytics import YOLO

# Step 1: Initialize the model
model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8 model

# Step 2: Train the model
model.train(
    data='config.yaml',
    epochs=10,
    imgsz=1280,
    batch=8,
)


# Step 3: Evaluate the model
results = model.val()

# Step 4: Export the model for inference
model.export(format='onnx')  # Export as ONNX model
