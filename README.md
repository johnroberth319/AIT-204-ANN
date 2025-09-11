# Handwritten Digit Recognition App

A Streamlit web application that recognizes handwritten digits (0-9) using a Convolutional Neural Network trained on the MNIST dataset.

## Features

- 📁 **Upload Images**: Upload images containing handwritten digits for recognition
- ✏️ **Draw Digits**: Draw digits directly in the browser using an interactive canvas
- 🎯 **Real-time Predictions**: Get instant predictions with confidence scores
- 📊 **Model Visualization**: View processed images and model predictions

## Demo

🚀 **[Try the live app on Streamlit Cloud](your-app-url-here)**

## Local Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/handwritten-digits.git
cd handwritten-digits
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (first time only):
```bash
python train_model.py
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure

```
handwritten-digits/
├── app.py                 # Main Streamlit application
├── train_model.py         # Script to train the MNIST model
├── predict.py            # Original prediction script
├── requirements.txt      # Python dependencies
├── mnist_model.h5       # Trained model (generated after training)
└── README.md            # This file
```

## Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Dataset**: MNIST (60,000 training images, 10,000 test images)
- **Input**: 28x28 grayscale images
- **Output**: Digit classification (0-9)
- **Accuracy**: ~99% on test set

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Deploy!

## Usage Tips

For best results:
- Use clear, single digits
- Ensure good contrast between digit and background
- Center the digit in the image
- Avoid multiple digits in one image

## Technologies Used

- **Streamlit**: Web application framework
- **TensorFlow/Keras**: Deep learning model
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **PIL**: Image handling

## License

This project is open source and available under the [MIT License](LICENSE).