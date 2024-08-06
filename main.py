import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for histogram equalization
import io
import pandas as pd
import logging
import os

# Create the logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging
logger = logging.getLogger('app_logger')
logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG to capture all events

file_handler = logging.FileHandler(os.path.join('logs', 'app.log'))
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Set to track logged messages
logged_messages = set()

# Helper function to log a message only once
def log_once(message, level=logging.INFO):
    if message not in logged_messages:
        logger.log(level, message)
        logged_messages.add(message)

# Log the application startup
log_once("Aplikasi dimulai")

# Define the class names
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Define the model architecture (must match the architecture used during training)
try:
    model_ft = models.resnet50(weights=None)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    log_once("Arsitektur model berhasil didefinisikan")
except Exception as e:
    log_once(f"Kesalahan saat mendefinisikan arsitektur model: {str(e)}", level=logging.ERROR)

# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the state_dict into the model with map_location to handle CPU
try:
    model_ft.load_state_dict(torch.load('brain_tumor_classification_model.pth', map_location=device))
    model_ft = model_ft.to(device)
    model_ft.eval()
    log_once("Model berhasil dimuat dan dipindahkan ke perangkat")
except Exception as e:
    log_once(f"Kesalahan saat memuat model: {str(e)}", level=logging.ERROR)

# Define data transforms (must match the transforms used during training)
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
log_once("Transformasi data berhasil didefinisikan")

# Function to apply histogram equalization and change color to green
def process_image(img, img_size=(256, 256)):
    try:
        # Convert to numpy array
        img_np = np.array(img)
        
        # Resize image
        img_resized = cv2.resize(img_np, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        if len(img_resized.shape) == 3:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_resized
        
        # Apply histogram equalization using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_eq = clahe.apply(img_gray)
        
        # Convert back to RGB
        img_eq_rgb = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)
        
        log_once("Gambar berhasil diproses (penyamaan histogram dan perubahan warna)")
        
        return Image.fromarray(img_eq_rgb), img_gray, img_eq
    except Exception as e:
        log_once(f"Kesalahan saat memproses gambar: {str(e)}", level=logging.ERROR)
        raise

# Function to plot histograms with clearer colors
def plot_histograms(img_gray, img_eq):
    try:
        # Calculate histograms
        hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])
        
        # Plot histograms
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(hist_gray, color='blue', label='Histogram Asli')
        ax.plot(hist_eq, color='lime', label='Histogram yang Diperbaiki')  # Clearer color
        ax.set_title('Perbandingan Histogram')
        ax.set_xlabel('Nilai Pixel')
        ax.set_ylabel('Frekuensi')
        ax.legend()
        
        # Save the plot to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        log_once("Plot perbandingan histogram berhasil dibuat")
        
        return buf
    except Exception as e:
        log_once(f"Kesalahan saat membuat plot histogram: {str(e)}", level=logging.ERROR)
        raise

# Function to plot accuracy and class distribution
def plot_accuracy_and_distribution(prediction_probs):
    try:
        # Example data for accuracy plot
        epochs = [1, 2, 3, 4, 5]
        accuracy = [0.60, 0.65, 0.70, 0.75, 0.80]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy Plot
        ax1.plot(epochs, accuracy, marker='o', linestyle='-', color='green')
        ax1.set_title('Akurasi Model Selama Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Akurasi')
        
        # Class Distribution Plot
        class_names_sorted = sorted(class_names)
        probs_sorted = [prediction_probs.get(name, 0) for name in class_names_sorted]
        
        ax2.bar(class_names_sorted, probs_sorted, color='cyan')
        ax2.set_title('Distribusi Kelas Prediksi')
        ax2.set_xlabel('Kelas')
        ax2.set_ylabel('Probabilitas')
        
        # Save the plot to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        log_once("Plot akurasi dan distribusi kelas berhasil dibuat")
        
        return buf
    except Exception as e:
        log_once(f"Kesalahan saat membuat plot akurasi dan distribusi kelas: {str(e)}", level=logging.ERROR)
        raise

def imshow_with_prediction(img, prediction):
    try:
        # Convert tensor to numpy image
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Display the image
        ax.imshow(img)
        ax.set_title(f'Prediksi: {prediction}', fontsize=20)
        ax.axis('on')
        
        # Get the dimensions of the image
        height, width, _ = img.shape
        
        # Annotate image with pixel size
        ax.text(0.5, -0.05, f'Lebar: {width}px, Tinggi: {height}px', 
                transform=ax.transAxes, ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        
        # Save the plot to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        log_once(f"Gambar ditampilkan dengan prediksi: {prediction}")
        
        return buf
    except Exception as e:
        log_once(f"Kesalahan saat menampilkan gambar dengan prediksi: {str(e)}", level=logging.ERROR)
        raise

# Nama pada bar browser
st.set_page_config(page_title="Klasifikasi Tumor Otak - Brain Imaging Tech")

# Streamlit app
st.title('Klasifikasi Tumor Otak')

# Copyright Web
st.markdown("""
---
&copy; 2024 Brain Imaging Tech. Hak Cipta Dilindungi.
""")

# Sidebar for navigation
st.sidebar.header("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Upload File", "Hasil Histogram", "Hasil Output Gambar"])

# State variables for storing uploaded image and prediction results
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'img_transformed' not in st.session_state:
    st.session_state.img_transformed = None
if 'prediction_probs' not in st.session_state:
    st.session_state.prediction_probs = None
if 'img_gray' not in st.session_state:
    st.session_state.img_gray = None
if 'img_eq' not in st.session_state:
    st.session_state.img_eq = None

if page == "Upload File":
    st.header("Unggah dan Proses Gambar")
    uploaded_file = st.file_uploader("Pilih gambar...", type="jpg")
    
    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.write("File berhasil diunggah")
        log_once("File berhasil diunggah")

        # Display the original image
        st.image(st.session_state.uploaded_image, caption='Gambar yang diunggah', use_column_width=True)

        # Apply histogram equalization and change color to green
        try:
            img_processed, img_gray, img_eq = process_image(st.session_state.uploaded_image)
            st.session_state.img_transformed = img_processed
            st.session_state.img_gray = img_gray
            st.session_state.img_eq = img_eq
            st.write("Gambar telah diproses")
            log_once("Gambar berhasil diproses")
        except Exception as e:
            st.write(f"Kesalahan saat memproses gambar: {str(e)}")
            log_once(f"Kesalahan saat memproses gambar: {str(e)}", level=logging.ERROR)
        
        # Plot histograms
        try:
            hist_buf = plot_histograms(st.session_state.img_gray, st.session_state.img_eq)
            st.image(hist_buf, caption="Perbandingan Histogram", use_column_width=True)
        except Exception as e:
            st.write(f"Kesalahan saat membuat plot histogram: {str(e)}")
            log_once(f"Kesalahan saat membuat plot histogram: {str(e)}", level=logging.ERROR)
        
        # Predict the class
        try:
            image_tensor = data_transforms(st.session_state.uploaded_image).unsqueeze(0).to(device)
            outputs = model_ft(image_tensor)
            _, preds = torch.max(outputs, 1)
            pred_class = class_names[preds.item()]
            st.session_state.prediction = pred_class
            st.session_state.prediction_probs = {name: float(outputs[0, i]) for i, name in enumerate(class_names)}
            st.write(f"Prediksi: {pred_class}")
            log_once(f"Prediksi dibuat: {pred_class}")
        except Exception as e:
            st.write(f"Kesalahan saat membuat prediksi: {str(e)}")
            log_once(f"Kesalahan saat membuat prediksi: {str(e)}", level=logging.ERROR)
            
        # Plot prediction and accuracy
        try:
            pred_img_buf = imshow_with_prediction(image_tensor[0].cpu(), st.session_state.prediction)
            st.image(pred_img_buf, caption="Gambar dengan Prediksi", use_column_width=True)
            
            acc_dist_buf = plot_accuracy_and_distribution(st.session_state.prediction_probs)
            st.image(acc_dist_buf, caption="Distribusi Kelas dan Akurasi", use_column_width=True)
        except Exception as e:
            st.write(f"Kesalahan saat membuat plot prediksi dan akurasi: {str(e)}")
            log_once(f"Kesalahan saat membuat plot prediksi dan akurasi: {str(e)}", level=logging.ERROR)

elif page == "Hasil Histogram":
    if st.session_state.img_transformed:
        st.header("Hasil Histogram")
        # Plot histograms
        try:
            hist_buf = plot_histograms(st.session_state.img_gray, st.session_state.img_eq)
            st.image(hist_buf, caption="Perbandingan Histogram", use_column_width=True)
        except Exception as e:
            st.write(f"Kesalahan saat membuat plot histogram: {str(e)}")
            log_once(f"Kesalahan saat membuat plot histogram: {str(e)}", level=logging.ERROR)
    else:
        st.write("Belum ada gambar yang diproses. Silakan unggah gambar di halaman Upload File.")
        log_once("Mencoba melihat histogram tanpa gambar yang diproses")

elif page == "Hasil Output Gambar":
    if st.session_state.prediction:
        st.header("Hasil Output Gambar")
        # Plot prediction and accuracy
        try:
            pred_img_buf = imshow_with_prediction(data_transforms(st.session_state.uploaded_image).unsqueeze(0).to(device)[0].cpu(), st.session_state.prediction)
            st.image(pred_img_buf, caption="Gambar dengan Prediksi", use_column_width=True)
            
            acc_dist_buf = plot_accuracy_and_distribution(st.session_state.prediction_probs)
            st.image(acc_dist_buf, caption="Distribusi Kelas dan Akurasi", use_column_width=True)
        except Exception as e:
            st.write(f"Kesalahan saat membuat plot prediksi dan akurasi: {str(e)}")
            log_once(f"Kesalahan saat membuat plot prediksi dan akurasi: {str(e)}", level=logging.ERROR)
    else:
        st.write("Belum ada prediksi. Silakan unggah gambar di halaman Upload File.")
        log_once("Mencoba melihat output tanpa prediksi")
