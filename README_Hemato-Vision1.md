
# Hemato-Vision

A web-based application for automated blood cell classification, designed to predict the type of blood cells (eosinophil, lymphocyte, monocyte, neutrophil) using a pre-trained deep learning model.

## Overview
Hemato-Vision is a machine learning application developed to assist in hematological analysis by classifying blood cell images. It uses a TensorFlow deep learning model (`Blood_Cell.h5`) trained on labeled cell images. The app is built with Flask and uses OpenCV for image handling. The goal is to create a simple, effective diagnostic support tool for educational and research purposes.

## Features
- Upload and classify blood cell images in real-time.
- Predicts one of the four classes: eosinophil, lymphocyte, monocyte, neutrophil.
- Displays predictions with confidence score.
- Clean, responsive UI with custom HTML and CSS.
- Works locally on any machine with Python installed.

## Setup Instructions

### 1. Prerequisites
- Python 3.12 or above recommended
- Git (to clone the repo)

### 2. Clone the Repository
```bash
git clone https://github.com/Allen015/Hemato-Vision1.git
cd Hemato-Vision1
```

### 3. Create and Activate Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Flask App
```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser to use the app.

## How to Use
1. Launch the app and go to the prediction page.
2. Upload a blood cell image (JPEG/PNG).
3. Click "Predict" and the system will classify the image.
4. The result will display the cell type and prediction confidence.

## Project Structure
```
Hemato-Vision1/
│
├── static/               # CSS, images
│   └── assets/           # Backgrounds, icons, etc.
│
├── templates/            # HTML templates (home.html, result.html)
│
├── Blood_Cell.h5         # Trained TensorFlow model (not included in repo due to size)
├── app.py                # Main Flask application
├── requirements.txt      # Dependency list
└── README.md
```

## Tech Stack
- **Backend:** Flask
- **Frontend:** HTML, CSS (Bootstrap)
- **Image Processing:** OpenCV (`cv2`)
- **ML Framework:** TensorFlow / Keras
- **Language:** Python 3.12+

## Model Details
- Model Name: `Blood_Cell.h5`
- Accuracy: ~92% on validation data
- Trained on publicly available blood cell datasets

## Known Issues
- The `Blood_Cell.h5` file is too large for GitHub upload (over 100 MB). You must download it separately or compress if needed.
- App only accepts `.jpg`, `.jpeg`, `.png` formats.

## Future Improvements
- Add support for mobile uploads.
- Enable batch image prediction.
- Deploy online via Render or Hugging Face Spaces.
- Add a feedback loop for incorrect predictions.

## License
All rights reserved by Allen015. Consider adding an open-source license for broader contribution and use.

## Credits
- Built using open-source libraries: Flask, TensorFlow, OpenCV
- Special thanks to mentors and peers who supported development.
