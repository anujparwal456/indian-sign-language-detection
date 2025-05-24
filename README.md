# Indian Sign Language Detection

This project is a machine learning-based application designed to detect and recognize **Indian Sign Language (ISL)** gestures — including both alphabets (`a-z`) and digits (`0-9`) — in real-time using a webcam. It uses computer vision techniques and a trained deep learning model via TensorFlow and OpenCV.

---

## 🚀 Features

- ✋ **Real-Time Gesture Recognition** using webcam
- 🧠 **Deep Learning CNN model** trained on ISL gestures
- 🖼️ **Custom Dataset Support** for letters and digits
- ⚙️ **Preprocessing pipeline**: Cropping, resizing, and normalization
- 🧪 **Model training, evaluation**, and visualization included
- 🌐 **Streamlit-based UI** for interactive gesture-to-text conversion

---

## 📁 Folder & File Structure

### 📂 Dataset

> Dataset folders are assumed to follow the structure:
- `Final_Sign_Dataset/` – raw gesture images, one subfolder per class
- `preprocessed_alphaDigi_dataset/` – auto-generated split dataset (train/val/test)

> _Note: Datasets should be placed manually. Not included due to size._

### 📓 Notebooks

- `dataCollection.ipynb` – Capture and save images via webcam
- `modelTraining.ipynb` – Train CNN model on the dataset
- `testingOpenCV.ipynb` – Live testing using OpenCV and webcam

### 📂 Scripts

- `modelTraining.py` – Full script version for training the model
- `appUI.py` – Main **Streamlit app** for real-time ISL recognition
- `sign_language_alphaDigi_model.h5` – Pretrained CNN model for classification
  
---
### Application

- **`appUI.py`**: Streamlit-based application for real-time gesture recognition and text generation.
- Use the terminal inside VS Code and run the following command: **streamlit run appUI.py**

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anujparwal456/indian-sign-language-detection.git
   cd indian-sign-language-detection
   ```
# Usage

## 1. Data Collection

To collect hand gesture images for training:

Run the dataCollection.ipynb notebook.
Use the webcam to capture images of hand gestures for each class (e.g., a, b, c, etc.).
Press s to save an image and q to quit.

## 2. Model Training

To train the model:

Run the modelTraining.ipynb notebook.
The dataset will be split into training, validation, and test sets.
The trained model will be saved as sign_language_alphaDigi_model.h5.

## 3. Real-Time Testing

To test the model in real-time:

```bash
   streamlit run appUI.py
```
Run the appUI.py script:
Use the Streamlit interface to open the camera and start recognizing gestures.

# Model Architecture

The model is a Convolutional Neural Network (CNN) with the following layers:

Convolutional Layers: Extract spatial features from images.
MaxPooling Layers: Reduce spatial dimensions.
Batch Normalization: Normalize activations for faster convergence.
Dropout: Prevent overfitting.
Dense Layers: Fully connected layers for classification.

## Requirements

Python 3.8+
OpenCV
TensorFlow
Streamlit
cvzone
NumPy
Matplotlib


---
# Sample Images 
---
## Images of Number 0,1,2
![image](https://github.com/user-attachments/assets/209b9895-a2c8-4607-a924-ef4931fd0dd9)
![image](https://github.com/user-attachments/assets/b63b165f-bdce-4f66-9a6c-0d8ee267dbec)
![image](https://github.com/user-attachments/assets/e4c3b3c5-66a6-49e2-9e29-e7b3e06b6da3)

---
## Images of Alphabet A,B,C
![image](https://github.com/user-attachments/assets/5345dbf7-f27e-4650-b3ed-2f1d1242ecab)
![image](https://github.com/user-attachments/assets/ed49c2d4-c558-41de-bd72-c4653e402b1a)
![image](https://github.com/user-attachments/assets/9775d77e-3001-490d-a833-9bf8eda8b4e4)

---
## ✅ To Use This

1. Save the content above in a file named `README.md` inside your project root folder.
2. Then run:

```bash
git add README.md
git commit -m "Add project README"
git push
----
