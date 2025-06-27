# Hemato-Vision - Project Documentation

## 1. Introduction

- **Project Title:** Hemato-Vision  
- **Team Members:**  
  - G. Allen Noah Bhagyam
  - L. Mounika
  - T. Malik Baba
  - N. Mahesh
  - P. Paramesh

## 2. Project Overview

- **Purpose:**  
  Hemato-Vision is an AI-based web application built to automate the classification of human blood cells using image inputs. This project aids medical diagnostics and hematology research by offering fast, accurate cell-type classification.

- **Features:**  
  - Upload blood cell image and get instant prediction  
  - Identifies four major cell types: eosinophil, lymphocyte, monocyte, neutrophil  
  - Displays prediction confidence  
  - Simple and clean web UI built with Flask  
  - Pre-trained deep learning model (`Blood_Cell.h5`)  
  - Real-time inference with OpenCV preprocessing

## 3. Architecture

- **Frontend:**  
  - HTML5, CSS3 (Bootstrap 5)  
  - Pages: Home, Predict, Result  
  - Responsive layout for desktop and mobile use

- **Backend:**  
  - Flask (Python micro web framework)  
  - Image upload handling  
  - Prediction using pre-trained model

- **Model (AIML Layer):**  
  - TensorFlow/Keras CNN model trained to classify blood cell images  
  - Image preprocessing via OpenCV  
  - Outputs class prediction and confidence score

- **Database:**  
  - No database used. Inference-only app without persistent storage.

## 4. Setup Instructions

- **Prerequisites:**  
  - Python 3.12  
  - pip  
  - Git (optional)  
  - Virtualenv (recommended)

- **Installation Steps:**  
  ```bash
  git clone https://github.com/Allen015/Hemato-Vision1.git
  cd Hemato-Vision1
  python -m venv venv
  venv\\Scripts\\activate     # On Windows
  pip install -r requirements.txt
  ```

- **Model File:**  
  Download or place the `Blood_Cell.h5` model file in the project root. (Not included in GitHub due to size restrictions.)

## 5. Folder Structure

```
Hemato-Vision1/
├── static/              # CSS and asset files
│   └── assets/          # Icons, background images
├── templates/           # HTML templates (home.html, result.html)
├── app.py               # Flask application
├── Blood_Cell.h5        # Trained TensorFlow model
├── requirements.txt     # Required Python libraries
└── README.md
```

## 6. Running the Application

```bash
python app.py
```

- Open a browser and go to `http://127.0.0.1:5000/`  
- Upload a blood cell image and view the prediction

## 7. API Documentation

| Endpoint   | Method | Description                        |
|------------|--------|------------------------------------|
| `/`        | GET    | Loads the home page                |
| `/predict` | GET    | Loads the upload form              |
| `/predict` | POST   | Receives image, returns prediction |

- **Example Response:**
```json
{
  "prediction": "lymphocyte",
  "confidence": 0.91
}
```

## 8. Authentication

- Not implemented. The application is fully public and runs locally without login or user roles.

## 9. User Interface

- Built with Bootstrap for responsiveness  
- Three main pages:
  - Home: App introduction and navigation
  - Predict: Upload interface for images
  - Result: Shows prediction result and confidence

## 10. Testing

- Manual testing conducted with multiple cell images  
- Verified model output and UI behavior  
- Tools used: Visual Studio Code (Python), Browser testing (Chrome)

## 11. Screenshots or Demo

![Project](https://github.com/user-attachments/assets/df98680b-7037-4feb-9c3e-fd455b239ba9)

-Demo Link: https://youtu.be/eP7AXKyhJg4

## 12. Known Issues

- `Blood_Cell.h5` is over 100MB and cannot be pushed to GitHub  
- No image validation before upload (e.g., size/resolution/type check)

## 13. Future Enhancements

- Add drag-and-drop image upload  
- Enable batch predictions  
- Model confidence heatmaps  
- Deploy using Render, Hugging Face Spaces, or Streamlit  
- Add historical logging/database for predictions  
- Improve dataset diversity for better accuracy
