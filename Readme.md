# YOLOv8 Object Detection App

A web application for object detection built using **YOLOv8 and Streamlit**. The application uses a pre-trained custom model (`best.pt`) to detect objects in uploaded images and display bounding boxes, class labels, and confidence scores.

## Features

- Object detection using a custom-trained YOLOv8 model.
- Image upload support for JPG, JPEG, and PNG formats.
- Real-time inference on uploaded images.
- Bounding boxes, class labels, and confidence scores.
- Adjustable confidence threshold for filtering detections.
- Detection summary with object counts and confidence scores.
- Visualization of detected object classes.
- Downloadable images with detection results.

## Tech Stack

- **Python** – Core programming language.
- **YOLOv8 (Ultralytics)** – Object detection model.
- **PyTorch** – Deep learning framework.
- **Streamlit** – Interactive web interface.
- **Pillow (PIL)** – Image processing.
- **Pandas** – Detection results and data visualization.

## Project Structure

```text
yolo-streamlit-app/
├── app.py
├── requirements.txt
└── models/
    └── best.pt
```

## Project Workflow

1. Prepare and annotate the dataset in YOLO format.
2. Train the YOLOv8 model on the custom dataset.
3. Save the trained model weights as `best.pt`.
4. Load the trained model using the Ultralytics YOLO library.
5. Upload an image through the Streamlit interface.
6. Run inference and identify objects in the image.
7. Display bounding boxes, class labels, and confidence scores.
8. Download the annotated image for further use.

## Installation and Setup

**1. Clone the repository**

```bash
git clone <your-repository-url>
cd yolo-streamlit-app
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add the trained model**

Place your trained `best.pt` file inside the `models/` directory.

**4. Run the application**

```bash
streamlit run app.py
```

The application will open in your default browser, usually at `http://localhost:8501`.

## Model Information

The application uses custom-trained YOLOv8 weights stored in `models/best.pt`. Detection performance depends on the training dataset, model configuration, image quality, and confidence threshold.

## Future Improvements

- Support for video object detection.
- Webcam-based real-time detection.
- Detection history and downloadable reports.
- Cloud deployment for public access.
- Performance optimization for GPU inference.
