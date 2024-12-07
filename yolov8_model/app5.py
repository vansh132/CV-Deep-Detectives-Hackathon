from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import tensorflow as tf

app = FastAPI()
output_dir = os.path.join('.', 'output_images')

# Path to the TensorFlow Lite model
model_path = 'best_float16.tflite'  # Replace with your actual model path

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Lower the threshold for detection
threshold = 0.5

@app.get('/')
def res():
    return {'message': "App5: Server running on PORT 127.0.0.1"}

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    try:
        # Convert file to an OpenCV image
        contents = await image.read()
        in_memory_file = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to load image")

        # Preprocess the image to match input shape
        input_shape = input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
        input_data = cv2.resize(img, (width, height))
        input_data = np.expand_dims(input_data, axis=0)
        input_data = input_data.astype(np.float32) / 255.0  # Normalize if necessary

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Perform inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        detection_made = False
        (im_height, im_width) = img.shape[:2]

        for i in range(output_data.shape[2]):
            ymin, xmin, ymax, xmax, score = output_data[0, :, i]
            if score > threshold:
                detection_made = True
                xmin, xmax, ymin, ymax = xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height

                # Draw bounding box on the image
                data =  {
                    "ymin": ymax, 
                    "ymax": ymin, 
                    "xmin": xmin, "xmax": xmax, }
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 4)
                cv2.putText(img, f"Score: {score:.2f}", (int(xmin), int(ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                
                # Save the processed image to a buffer (optional, if you want to save the image for debugging)
        output_path = os.path.join(output_dir, 'processed_image.jpg')
        cv2.imwrite(output_path, img)

        message = "Logo detected" if detection_made else "Logo not detected"
        return JSONResponse(content={"status": detection_made, "message": message, "data": data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, "127.0.0.2", port=9000)