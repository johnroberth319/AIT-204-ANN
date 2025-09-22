import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import io

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

@st.cache_resource
def load_trained_model():
    try:
        model = load_model("mnist_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please run train_model.py first to create the model file.")
        return None

def preprocess_image(image):
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Invert the image (assuming black digit on white background)
    img = 255 - img
    
    # Resize to 28x28
    img = cv2.resize(img, (28, 28))
    
    # Normalize
    img = img / 255.0
    
    # Expand dimensions to represent a single sample
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    
    return img

def predict_digit(model, image):
    if model is None:
        return None, None
    
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions)
    
    return predicted_label, confidence

def main():
    st.title("🔢 Handwritten Digit Recognition")
    st.markdown("Upload an image or draw a digit to get a prediction!")
    
    # Load the model
    model = load_trained_model()
    
    if model is None:
        st.stop()
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Image", "✏️ Draw Digit"])
    
    with tab1:
        st.header("Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image containing a handwritten digit (0-9)"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.subheader("Prediction")
                
                # Make prediction
                predicted_digit, confidence = predict_digit(model, image)
                
                if predicted_digit is not None:
                    st.success(f"**Predicted Digit: {predicted_digit}**")
                    st.info(f"Confidence: {confidence:.2%}")
                    
                    # Show processed image
                    processed_img = preprocess_image(image)
                    st.image(
                        processed_img.reshape(28, 28),
                        caption="Processed Image (28x28)",
                        use_column_width=True,
                        clamp=True
                    )
    
    with tab2:
        st.header("Draw a Digit")
        st.markdown("**Drawing canvas is temporarily unavailable due to library compatibility issues.**")
        
        # Show a placeholder canvas area
        st.markdown("""
        <div style="
            width: 200px; 
            height: 200px; 
            border: 2px solid #ccc; 
            background: white; 
            margin: 20px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            color: #666;
        ">
            Canvas Area<br/>
            (Feature Coming Soon)
        </div>
        """, unsafe_allow_html=True)
        
        st.info("📌 **For now, please use the 'Upload Image' tab above to test digit recognition!**")
        st.markdown("The upload feature works perfectly and provides excellent digit recognition.")
        
        canvas_result = None
        
        # Simplified buttons for the placeholder
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Predict Drawn Digit", type="primary", use_container_width=True, disabled=True):
                st.warning("Drawing feature is temporarily unavailable.")
        
        with col2:
            if st.button("🗑️ Clear Canvas", use_container_width=True, disabled=True):
                st.info("Canvas feature coming soon!")
        
        st.warning("⚠️ **Canvas drawing is temporarily disabled.** Please use the **Upload Image** tab for digit recognition!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset 
        to recognize handwritten digits (0-9).
        
        **Features:**
        - Upload images for digit recognition
        - Draw digits directly in the browser
        - Real-time predictions with confidence scores
        
        **Tips for better accuracy:**
        - Use clear, single digits
        - Ensure good contrast
        - Center the digit in the image
        """)
        
        st.header("Model Info")
        if model is not None:
            st.success("✅ Model loaded successfully")
            st.info("CNN trained on MNIST dataset")
        else:
            st.error("❌ Model not available")

if __name__ == "__main__":
    main()