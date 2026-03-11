# Air Writing Detector – Math Character Recognition

Real-time computer vision system that recognizes mathematical expressions written in the air using hand gestures. Built with OpenCV, MediaPipe, and TensorFlow, it tracks finger trajectories, detects symbols with a CNN model, and evaluates equations through a gesture-controlled interface.

A modular Python project that trains a CNN to recognise handwritten mathematical characters from **28 × 28 grayscale images**. The trained model can later be integrated with MediaPipe hand-tracking to build a **real-time Air Writing Detector** that converts finger trajectories directly into text.

---

## Project Structure

```text
Air_writing_detector/
├── src/
│   └── main.py                              – latest working inference code
├── training/
│   └── Air_writing_detector_training.py     – end-to-end training script
├── model/
│   └── scientific_model1.h5                 – trained math evaluation weights
├── assets/                                  – images used in README
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Installation & Running

Follow these steps to set up and run the Air Writing Detector:

```bash
git clone https://github.com/Navdeep-2510/Air_writing_detector.git
cd Air_writing_detector
pip install -r requirements.txt
python src/main.py
```

**Note:**
The `python src/main.py` script automatically loads the `scientific_model1.h5` model located in the `model/` directory for mathematical expression recognition.

---

## Training (Optional)

If you want to train the model from scratch, run the training script:

```bash
python training/Air_writing_detector_training.py
```

The training pipeline:

• Loads MNIST digit dataset
• Generates synthetic mathematical symbols
• Trains a CNN classifier
• Saves the trained model to the `model/` folder

The final model is saved as:

```
model/scientific_model1.h5
```

---

## Next Steps

* Integrate with **MediaPipe hand landmark tracking**
* Build the **real-time canvas / stroke accumulation system**
* Add **advanced algebra parsing and evaluation**
* Improve recognition accuracy with larger datasets

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* NumPy

---

## License

This project is licensed under the MIT License.

