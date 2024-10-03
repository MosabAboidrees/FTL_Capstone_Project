import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pywt
import cv2
from tensorflow.keras.models import load_model

# Display the project title and author information at the top
st.title("ECG Classification based on Time-Frequency Analysis and Deep Learning")
st.markdown("""
**By:** Mosab Aboidrees Altraifi Yousif and Abdelrahman Mohammed (Group 12)
""")

# Provide an introduction and explanation of the system
st.markdown("""
## Overview
This system analyzes ECG signals using Continuous Wavelet Transform (CWT) and a pre-trained Convolutional Neural Network (CNN) model. It allows you to upload an ECG signal file in `.npy` format, visualize the signal, generate its time-frequency representation (scalogram) using CWT, and classify the signal into one of the following categories:
- **ARR**: Arrhythmia
- **CHF**: Congestive Heart Failure
- **NSR**: Normal Sinus Rhythm

### How to Use:
1. **Upload the Signal**: Click the 'Browse files' button below to upload an ECG signal in `.npy` format from your computer.
2. **View the Signal**: Once uploaded, the first 1000 samples of the ECG signal will be plotted.
3. **Apply CWT**: The system will apply Continuous Wavelet Transform (CWT) to the entire ECG signal and display a scalogram.
4. **Predict the Class**: The system will use a pre-trained CNN model to predict whether the ECG signal belongs to the ARR, CHF, or NSR class.

### Note:
Make sure the uploaded file is a `.npy` file containing the ECG signal data.
""")

# Section: Upload the Signal
st.markdown("---")
st.markdown("### 1. Upload the Signal")
uploaded_file = st.file_uploader("Choose an ECG signal file (.npy)", type="npy")

if uploaded_file is not None:
    # Load the ECG signal from the uploaded .npy file
    st.write("Loading ECG signal...")
    signal = np.load(uploaded_file)

    # Section: Plotting the ECG Signal
    st.markdown("---")
    st.markdown("### 2. Plotting the ECG Signal")
    
    # Plot the first 1000 samples of the ECG signal
    signal_to_plot = signal[:1000]
    st.write("The following plot shows the first 1000 samples of the ECG signal:")
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(signal_to_plot)), signal_to_plot)
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Amplitude')
    ax.set_title('ECG Signal')
    st.pyplot(fig)

    # Section: Applying Continuous Wavelet Transform (CWT)
    st.markdown("---")
    st.markdown("### 3. Applying Continuous Wavelet Transform (CWT)")
    st.write("""
    Continuous Wavelet Transform (CWT) is a powerful tool for analyzing time-frequency characteristics of signals like ECG. 
    By using a wavelet (in this case, the 'morl' wavelet), we can decompose the ECG signal into various frequency components while retaining the time information. 
    This results in a **scalogram**, a 2D time-frequency representation of the signal.
    """)

    st.write("Applying Continuous Wavelet Transform to the entire ECG signal...")
    sampling_frequency = 128
    scales = np.arange(1, 129)
    coef, freqs = pywt.cwt(signal, scales=scales, wavelet='morl', sampling_period=1/sampling_frequency)
    scalogram = np.abs(coef)

    # Resize and normalize the scalogram
    scalogram_resized = cv2.resize(scalogram, (224, 224))
    scalogram_resized = (scalogram_resized - np.min(scalogram_resized)) / (np.max(scalogram_resized) - np.min(scalogram_resized))

    # Plot the scalogram
    st.write("The following scalogram represents the time-frequency distribution of the ECG signal:")
    fig, ax = plt.subplots()
    im = ax.imshow(scalogram, cmap='jet', aspect='auto', extent=[0, len(signal), freqs[-1], freqs[0]])
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Continuous Wavelet Transform (CWT) of ECG Signal')
    plt.colorbar(im, ax=ax)  # Add colorbar with the mappable `im`
    st.pyplot(fig)

    # Section: Prediction
    st.markdown("---")
    st.markdown("### 4. CNN Prediction of ECG Signal Class")
    st.write("""
    The system uses a pre-trained Convolutional Neural Network (CNN) model to classify the ECG signal. 
    The CNN has been trained on similar ECG data and can predict whether the signal belongs to one of the following categories:
    - **ARR**: Arrhythmia
    - **CHF**: Congestive Heart Failure
    - **NSR**: Normal Sinus Rhythm
    """)

    # Load the saved CNN model and make a prediction
    st.write("Loading the trained CNN model and making a prediction...")
    model_path = 'ecg_cnn_model.h5'  # Replace with your actual model path
    loaded_model = load_model(model_path)

    # Prepare the scalogram for the model
    scalogram_resized = np.expand_dims(scalogram_resized, axis=0)
    scalogram_resized = np.expand_dims(scalogram_resized, axis=-1)

    # Make a prediction
    prediction = loaded_model.predict(scalogram_resized)
    predicted_class = np.argmax(prediction)

    # Map the predicted class to its label
    label_mapping = {0: 'ARR', 1: 'CHF', 2: 'NSR'}
    predicted_label = label_mapping.get(predicted_class, 'Unknown')

    # Display the prediction result
    st.markdown(f"#### The predicted label is: **{predicted_label}**")

else:
    # Display a message when no file is uploaded
    st.write("Please upload an ECG signal file in .npy format to start the analysis.")
