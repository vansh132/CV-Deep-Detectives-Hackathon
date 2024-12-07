from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr

app = FastAPI()
reader = easyocr.Reader(['en'])
output_dir = os.path.join('.', 'output_images')
os.makedirs(output_dir, exist_ok=True)

# Path to the trained model
model_path = os.path.join('.', 'runs', 'detect', 'train6', 'weights', 'last.pt')

# Load the custom model
model = YOLO(model_path)

# Lower the threshold for detection
threshold = 0.3
output_dir = "/Users/vanshah/Documents/yolo/yolov8/output_images/"


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
                
                # Crop the detected bounding box region
                cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
                cropped_images.append(cropped_img)

                # Save the cropped image to a file
                cropped_img_path = os.path.join(output_dir, f'cropped_{int(x1)}_{int(y1)}.jpg')
                cv2.imwrite(cropped_img_path, cropped_img)

                 # Perform text recognition on the cropped image
                text_recognition_results = reader.readtext(cropped_img)
                recognized_text = " ".join([text for bbox, text, score in text_recognition_results])
                print(recognized_text)
                if "deemed" in recognized_text.lower():
                    return JSONResponse(content={"message": "Valid poster"})
                else:
                    return JSONResponse(content={"message": "Invalid poster"})
        
            else:
                print(f"Detection below threshold: score = {score}, class = {results.names[int(class_id)]}")

        # Save the processed image to a buffer (optional, if you want to save the image for debugging)
        output_path = os.path.join(output_dir, 'processed_image.jpg')
        cv2.imwrite(output_path, img)

        message = "Detection made" if detection_made else "No detection"

        return JSONResponse(content={"message": message})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/upload")
# async def upload_image(image: UploadFile = File(...)):
#     try:
#         # Convert file to an OpenCV image
#         contents = await image.read()
#         in_memory_file = np.frombuffer(contents, np.uint8)
#         img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

#         if img is None:
#             raise HTTPException(status_code=400, detail="Failed to load image")

#         results = model(img)[0]

#         # Debugging: Print results
#         # print(f"Detections: {results.boxes.data.tolist()}")

#         detection_made = len(results.boxes.data) > 0

#         for result in results.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = result

#             if score > threshold:
#                 cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#                 cv2.putText(img, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
#             else:
#                 print(f"Detection below threshold: score = {score}, class = {results.names[int(class_id)]}")

#         # Save the processed image to a buffer (optional, if you want to save the image for debugging)
#         output_path = os.path.join(output_dir, 'processed_image.jpg')
#         cv2.imwrite(output_path, img)

#         message = "Detection made" if detection_made else "No detection"

        

#         return JSONResponse(content={"message": message})

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

