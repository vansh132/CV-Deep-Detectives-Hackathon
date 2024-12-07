from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr

app = FastAPI()
reader = easyocr.Reader(['en'])
# output_dir = os.path.join('.', 'output_images')
# os.makedirs(output_dir, exist_ok=True)

# Path to the trained model
model_path = os.path.join('.', 'runs', 'detect', 'train6', 'weights', 'last.pt')
# model_path = os.path.join( 'yolov8n.pt')
# print(model_path)
# Load the custom model
model = YOLO(model_path)

# Lower the threshold for detection
threshold = 0.5

@app.get('/')
def res():
    return {'message': "Server running on PORT 127.0.0.1"}
    
@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    try:
        # Convert file to an OpenCV image
        contents = await image.read()
        in_memory_file = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (1280, 1280))

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to load image")

        results = model(img)[0]

        detection_made = len(results.boxes.data) > 0
        cropped_images = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                # Draw the bounding box and label on the image
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(img, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                print(f"y1 value: {y1}, y2 value: {y2}, x1 value: {x1}, x2 value: {x2}")
                # Crop the detected bounding box region
                cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                cropped_images.append(cropped_img)

                # Save the cropped image to a file
                # cropped_img_path = os.path.join(output_dir, f'cropped_{int(x1)}_{int(y1)}.jpg')
                # cv2.imwrite(cropped_img_path, cropped_img)

                # Perform text recognition on the cropped image
                text_recognition_results = reader.readtext(cropped_img)
                recognized_text = " ".join([text for bbox, text, score in text_recognition_results])
                print(recognized_text)
                if "deemed" in recognized_text.lower():
                    return JSONResponse(content={"status": True, "message": "Valid poster"})
                else:
                    return JSONResponse(content={"status": False, "message": "Invalid poster"})
        
            else:
                print(f"Detection below threshold: score = {score}, class = {results.names[int(class_id)]}")

        # Save the processed image to a buffer (optional, if you want to save the image for debugging)
        # output_path = os.path.join(output_dir, 'processed_image.jpg')
        # cv2.imwrite(output_path, img)

        message = "Logo detected" if detection_made else "Logo not detected"
       

        return JSONResponse(content={"status": False, "message": message})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
