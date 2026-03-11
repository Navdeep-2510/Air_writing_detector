# Air Writing Detector

A real-time computer vision system that recognizes mathematical expressions written in the air using hand gestures.
The system uses **MediaPipe hand tracking**, **OpenCV trajectory capture**, and a **CNN-based handwritten character recognition model** to detect symbols and evaluate mathematical expressions.

The project demonstrates gesture-based human–computer interaction by converting finger movements into digital text and mathematical expressions.

---

# Project Structure

```
Air_writing_detector
│
├── src
│   └── main.py
│       Main script for real-time air writing detection and expression evaluation
│
├── training
│   └── Air_writing_detector_training.py
│       Script used to train the CNN model
│
├── model
│   └── scientific_model1.h5
│       Trained CNN model used for symbol recognition
│
├── assets
│   Training results and evaluation outputs including:
│   - Training accuracy/loss graphs
│   - Confusion matrix
│   - Classification report
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# Features

• Real-time hand tracking using MediaPipe
• Air writing using finger trajectory detection
• CNN-based handwritten symbol recognition
• Mathematical expression evaluation
• Gesture-based interaction system
• Interactive dashboard displaying detected expressions and results

---

# Gesture Controls

Pinch (Thumb + Index Finger)
Start writing in the air

Victory Gesture ✌️
Save the detected expression

Fist ✊
Evaluate the saved expression

Thumbs Up 👍
Undo the last saved expression

Shake Hand
Clear the current writing

---

# Supported Symbols

Numbers
0 1 2 3 4 5 6 7 8 9

Operators

/ * - + 

Other Symbols
=  .  (  )  ^  √

Variables
x , y

Total supported classes: **22**

---

# Installation

Clone the repository:

```
git clone https://github.com/Navdeep-2510/Air_writing_detector.git
cd Air_writing_detector
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the project:

```
python src/main.py
```

---

# Model Training (Optional)

If you want to retrain the model from scratch:

```
python training/Air_writing_detector_training.py
```

The training script:

• Loads MNIST digits dataset
• Generates synthetic mathematical symbols
• Trains a CNN classifier
• Saves the trained model to:

```
model/scientific_model1.h5
```

---

# Training Results

The **assets/** folder contains model evaluation outputs including:

• Training accuracy and loss graphs
• Confusion matrix visualization
• Classification report for all classes

These artifacts help analyze model performance and recognition accuracy.

---

# Technologies Used

Python
OpenCV
MediaPipe
TensorFlow / Keras
NumPy

---

# License

This project is licensed under the MIT License.
