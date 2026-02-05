"""
streamlit_app.py - Web Demo for Hard Hat Detector

Streamlit is a Python library that makes it easy to create web apps
for machine learning projects. No HTML/CSS/JS required!

Run this app with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add src directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

import streamlit as st
from PIL import Image
import torch

from inference import HardHatPredictor


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Hard Hat Detector",
    page_icon="👷",
    layout="centered"
)


# =============================================================================
# LOAD MODEL (cached so it only loads once)
# =============================================================================

@st.cache_resource
def load_model():
    """Load the trained model. Cached to avoid reloading on every interaction."""
    model_path = Path(__file__).parent.parent / "models" / "best_model.pth"

    if not model_path.exists():
        return None

    return HardHatPredictor(str(model_path))


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.title("👷 Hard Hat Detector")
    st.markdown("""
    Upload an image to detect if a person is wearing a hard hat.

    This model uses a **ResNet18** neural network trained with transfer learning.
    """)

    # Load model
    predictor = load_model()

    if predictor is None:
        st.error("""
        **Model not found!**

        Please train the model first:
        ```bash
        cd src
        python train.py --data_dir ../data
        ```
        """)
        return

    # File uploader
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Make prediction
        with st.spinner("Analyzing..."):
            # Save uploaded file temporarily
            temp_path = Path("temp_upload.jpg")
            image.save(temp_path)

            # Get prediction
            result = predictor.predict(str(temp_path))

            # Clean up
            temp_path.unlink()

        # Display results
        with col2:
            st.subheader("Result")

            # Main prediction with emoji
            if result['class'] == 'hard_hat':
                st.success(f"✅ **HARD HAT DETECTED**")
            else:
                st.error(f"⚠️ **NO HARD HAT DETECTED**")

            # Confidence meter
            st.metric(
                label="Confidence",
                value=f"{result['confidence'] * 100:.1f}%"
            )

            # Probability breakdown
            st.subheader("Probabilities")
            for class_name, prob in result['probabilities'].items():
                st.progress(prob, text=f"{class_name}: {prob * 100:.1f}%")

    # Footer with info
    st.markdown("---")
    st.markdown("""
    ### About This Model

    - **Architecture**: ResNet18 with transfer learning
    - **Training**: Fine-tuned on hard hat detection dataset
    - **Classes**: Hard Hat, No Hard Hat

    ### How It Works

    1. **Upload**: Select an image containing a person
    2. **Preprocess**: Image is resized and normalized
    3. **Predict**: Neural network analyzes the image
    4. **Result**: Classification with confidence score
    """)


if __name__ == "__main__":
    main()
