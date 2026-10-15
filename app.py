
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import pandas as pd
import io

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🎯",
    layout="wide"
)

# -----------------------------
# Load the trained model
# -----------------------------
MODEL_PATH = Path("models/best.pt")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH.resolve()}"
        )

    return YOLO(str(MODEL_PATH))


# -----------------------------
# App header
# -----------------------------
st.title("🎯 YOLO Object Detection")
st.write(
    "Upload an image and let your trained YOLO model "
    "detect objects automatically."
)

# -----------------------------
# Sidebar settings
# -----------------------------
st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Higher values filter out less-confident detections."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Detection
# -----------------------------
if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Original Image",
            use_container_width=True
        )

        if st.button("Run Object Detection", type="primary"):

            with st.spinner("Detecting objects..."):

                model = load_model()

                results = model.predict(
                    source=image,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]

                # YOLO returns an annotated image in BGR format.
                # Convert it to RGB for Streamlit.
                annotated_image = result.plot()[:, :, ::-1]

                # -----------------------------
                # Results
                # -----------------------------
                st.success("Detection completed!")

                st.subheader("Detection Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        image,
                        caption="Original Image",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        annotated_image,
                        caption="Detected Objects",
                        use_container_width=True
                    )

                # -----------------------------
                # Detection statistics
                # -----------------------------
                boxes = result.boxes
                total_objects = len(boxes)

                st.subheader("Detection Summary")

                st.metric(
                    "Total Objects Detected",
                    total_objects
                )

                if total_objects > 0:

                    class_ids = (
                        boxes.cls.cpu().numpy().astype(int)
                    )
                    confidences = boxes.conf.cpu().numpy()

                    names = [
                        result.names[class_id]
                        for class_id in class_ids
                    ]

                    summary = pd.DataFrame({
                        "Class": names,
                        "Confidence": [
                            round(float(c), 4)
                            for c in confidences
                        ]
                    })

                    st.dataframe(
                        summary,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader("Class Counts")

                    counts = summary["Class"].value_counts()
                    st.bar_chart(counts)

                else:
                    st.info(
                        "No objects detected. "
                        "Try lowering the confidence threshold."
                    )

                # -----------------------------
                # Download annotated image
                # -----------------------------
                output_image = Image.fromarray(
                    annotated_image
                )

                buffer = io.BytesIO()
                output_image.save(buffer, format="PNG")

                st.download_button(
                    label="Download Detected Image",
                    data=buffer.getvalue(),
                    file_name="detected_image.png",
                    mime="image/png"
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload an image to get started.")
