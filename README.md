# 🎯 VisionHack 2024 - Event Poster Validation 🎨

## 📝 Problem Statement
The goal is to validate and verify event posters for **CHRIST (Deemed to be University)**, specifically ensuring the presence of the official university logo on each poster. ✅ This helps maintain branding consistency and ensures compliance with university guidelines.

---

## 💡 Solution
We developed a solution using **YOLOv8 Object Detection** and RESTful APIs to automate the poster validation process. 🚀

### 🎥 Demonstration & Explanation Video 
[Click here to watch the video](https://drive.google.com/file/d/1SiUYaB1ZbCDfc0IjLmtexmWu8vEq874J/view?usp=sharing)

### 🌟 Key Features:
1. **🎯 Custom Object Detection Model**: 
   - Trained a YOLOv8 model on a custom dataset of event posters with annotated logos.
   - Fine-tuned for high accuracy in detecting the official university logo. 🏅

2. **🔗 REST API Integration**:
   - Built REST APIs using [FastAPI](https://fastapi.tiangolo.com/) to integrate the trained model with the front-end.
   - Provided endpoints to upload posters and retrieve validation results. 🖼️

3. **🔍 API Testing**:
   - APIs thoroughly tested using [Postman](https://www.postman.com/) to ensure robust performance and reliability. 🛠️

---

## 🛠️ Tech Stack
- **📚 Frameworks and Libraries**:
  - [YOLOv8](https://github.com/ultralytics/yolov8) for object detection.
  - [FastAPI](https://fastapi.tiangolo.com/) for REST API development.
- **🔧 Tools**:
  - [Postman](https://www.postman.com/) for API testing.
  - Python 🐍 for scripting and model development.

## 🚀 How to Run the Project

Follow these steps to set up and run the project locally:

### Step 1: 📂 Download or Clone the Project
- Download the ZIP file from this repository and extract it, **or** 
- Clone the repository using the following command:
  ```bash
  git clone https://github.com/vansh132/poster-validation.git
  ```
   
### Step 2: 📂 Navigate to the Project Directory
- Open a terminal and navigate to the server folder:
  ```bash
  cd poster_validation_server
  ```

### Step 3: 📦 Install Dependencies
- Install the required Python dependencies using pip:
  ```bash
  pip3 install -r requirements.txt
  ```

### Step 4: ▶️ Run the Application
- Start the server by running the main script:
  ```bash
  python3 main.py
  ```


## 🔗 API Endpoints

### POST `/validate-poster`
- **Description**: Uploads a poster and returns validation results.

#### 📥 Request:
- **Body**: Form-data containing the poster image file.

#### 📤 Response:
- **Format**: JSON
- **Content**:
  - `status`: Validation status (e.g., `"success"` or `"error"`)
  - `detected_regions`: List of regions where the logo was detected, including coordinates.

#### Example Response:
```json
{
    "status": true,
    "message": "Logo detected",
    "bounding_boxes": [
        {
            "xmin": 946,
            "ymin": 27,
            "xmax": 1264,
            "ymax": 108,
            "score": 0.8638066649436951
        }
    ]
}
```

## 🙏 Acknowledgment

We would like to express our heartfelt gratitude to [**Dr. Helen K Joy**](mailto:helenk.joy@christuniversity.in) for providing us with this incredible opportunity to showcase our technical skills and explore new areas in Machine Learning. Your guidance, encouragement, and support have been invaluable throughout this journey. 

Thank you for inspiring us to push our boundaries and achieve excellence in this project.

## 👥 Contributors

- [**Riya Shah - 2347151**](https://www.linkedin.com/in/shahriyap/)  
- [**Vansh Shah - 2347152**](https://www.linkedin.com/in/vanshah/)  
