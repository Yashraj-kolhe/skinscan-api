# SkinScan AI 🔬 

SkinScan AI is a full-stack, cloud-integrated deep learning application designed for real-time dermatological screening. Built as a final-year engineering project, it leverages a Convolutional Neural Network (CNN) to analyze skin lesions and provide immediate, high-accuracy diagnostic insights.

## 🚀 Live Demo
**[SkinScan AI Live Dashboard](https://yashraj-kolhe.github.io/skinscan-api/)**

*(Note: The backend is hosted on a free Render instance and may take ~50 seconds to spin up on the very first request. Subsequent requests are processed in milliseconds).*

---

## 🧠 System Architecture

The application is built using a modern decoupled architecture:

* **Frontend (Presentation Layer):** A responsive, mobile-first web dashboard hosted globally on **GitHub Pages**. Features include a drag-and-drop image analyzer, an integrated Google Maps hospital locator for nearby dermatology centers, and skin cancer prevention guidelines.
* **Backend (Logic Layer):** A high-performance REST API built with **FastAPI (Python)**, containerized and deployed to **Render**.
* **AI Engine (Inference Layer):** A Custom CNN model utilizing Transfer Learning (MobileNetV2 architecture). The model was converted to **TensorFlow Lite (TFLite)** for highly optimized, low-latency edge inference. 

---

## ✨ Key Features

* **Real-Time Inference:** Processes clinical/dermoscopic images and returns a diagnosis alongside a statistical confidence score.
* **Advanced Pre-processing:** Utilizes **OpenCV** in the backend for automatic image resizing, normalization, and Gaussian noise reduction before model inference.
* **Healthcare Locator:** Integrated map module allowing users to find specialized oncology and dermatology centers in major cities.
* **Developer/Admin Console:** A hidden, built-in monitoring console that tracks session telemetry, real-time scanning statistics, and provides direct links to cloud server logs.
* **CORS & Cloud Security:** Strictly configured Cross-Origin Resource Sharing (CORS) to ensure the FastAPI backend only accepts payloads from the authenticated GitHub Pages frontend.

---

## 🛠️ Technology Stack

**Machine Learning & Computer Vision:**
* TensorFlow / Keras
* TensorFlow Lite (TFLite)
* OpenCV (cv2)
* NumPy & Pillow

**Backend / DevOps:**
* FastAPI (Python)
* Uvicorn (ASGI Server)
* Render (Cloud Hosting)
* Git / GitHub Version Control

**Frontend:**
* HTML5, CSS3, JavaScript (Vanilla)
* Browser LocalStorage API

---

## 👨‍💻 Developer
Developed by **Yashraj Kolhe** as a Final Year Engineering Project.
