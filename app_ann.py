import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import io

st.set_page_config(
    page_title="Handwritten Digit Recognition - ANN",
    page_icon="🔢",
    layout="wide"
)

@st.cache_resource
def load_trained_model():
    try:
        model = load_model("mnist_ann_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading ANN model: {e}")
        st.info("Please run train_ann_model.py first to create the ANN model file.")
        return None

def preprocess_image_for_ann(image):
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
    
    # Flatten for ANN (784 features)
    img_flat = img.reshape(1, 28 * 28)
    
    return img_flat, img

def predict_digit(model, image):
    if model is None:
        return None, None
    
    processed_img_flat, processed_img_2d = preprocess_image_for_ann(image)
    predictions = model.predict(processed_img_flat)
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions)
    
    return predicted_label, confidence, processed_img_2d

def main():
    st.title("🔢 Handwritten Digit Recognition - ANN Version")
    st.markdown("Upload an image to get predictions using an **Artificial Neural Network (ANN)**!")
    
    # Load the model
    model = load_trained_model()
    
    if model is None:
        st.stop()
    
    # Model info in sidebar
    with st.sidebar:
        st.header("🧠 ANN Model Info")
        st.success("✅ ANN Model loaded successfully")
        
        st.markdown("""
        **Architecture**: Artificial Neural Network
        - **Input**: 784 features (28×28 flattened)
        - **Hidden Layer 1**: 512 neurons + ReLU + Dropout
        - **Hidden Layer 2**: 256 neurons + ReLU + Dropout  
        - **Hidden Layer 3**: 128 neurons + ReLU + Dropout
        - **Output**: 10 classes (digits 0-9)
        
        **Key Differences from CNN**:
        - Flattens input images to 1D vectors
        - Uses fully connected layers only
        - No spatial feature extraction
        - Simpler but still effective for MNIST
        """)
    
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
                st.subheader("ANN Prediction")
                
                # Make prediction
                predicted_digit, confidence, processed_img = predict_digit(model, image)
                
                if predicted_digit is not None:
                    st.success(f"**Predicted Digit: {predicted_digit}**")
                    st.info(f"Confidence: {confidence:.2%}")
                    
                    # Show processed image
                    st.image(
                        processed_img,
                        caption="Processed Image (28x28 → flattened to 784 features)",
                        use_column_width=True,
                        clamp=True
                    )
                    
                    # Show flattening info
                    st.markdown(f"""
                    **ANN Processing**:
                    - Resized to 28×28 = 784 pixels
                    - Flattened to 1D vector of 784 features
                    - Fed to fully connected layers
                    """)
    
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
        
        st.info("📌 **For now, please use the 'Upload Image' tab above to test ANN digit recognition!**")
        st.markdown("The upload feature works perfectly and demonstrates ANN preprocessing.")
        
        # Simplified buttons for the placeholder
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Predict with ANN", type="primary", use_container_width=True, disabled=True):
                st.warning("Drawing feature is temporarily unavailable.")
        
        with col2:
            if st.button("🗑️ Clear Canvas", use_container_width=True, disabled=True):
                st.info("Canvas feature coming soon!")
        
        st.warning("⚠️ **Canvas drawing is temporarily disabled.** Please use the **Upload Image** tab for ANN digit recognition!")
    
    # Main sidebar with comparison info
    with st.sidebar:
        st.header("📊 CNN vs ANN")
        st.markdown("""
        **Comparison with CNN Version**:
        
        🔄 **CNN (Convolutional)**:
        - Preserves spatial relationships
        - Uses 2D convolutions + pooling
        - Fewer parameters, more efficient
        - Better for image tasks
        
        🧠 **ANN (Fully Connected)**:
        - Flattens images to vectors
        - Uses only dense layers
        - More parameters needed
        - Simpler architecture
        
        Both work well for MNIST digits!
        """)
        
        st.header("About")
        st.markdown("""
        This app demonstrates **Artificial Neural Network (ANN)** 
        for handwritten digit recognition.
        
        **Features:**
        - Upload images for digit recognition
        - ANN-specific preprocessing (flattening)
        - Real-time predictions with confidence scores
        - Architecture comparison with CNN
        
        **Tips for better accuracy:**
        - Use clear, single digits
        - Ensure good contrast
        - Center the digit in the image
        """)

if __name__ == "__main__":
    main()