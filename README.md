# Air Writing Detector – Math Character Recognition

A modular Python project that trains a CNN to recognise handwritten mathematical characters from 28 × 28 grayscale images. The trained model can later be integrated with MediaPipe hand-tracking to build a **real-time Air Writing Detector** that converts finger trajectories directly into text.

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

> **Note:** The `python src/main.py` script automatically utilizes the `scientific_model1.h5` model path located in the `model/` directory for mathematical expression inferences.

---

## Training (Optional)

If you want to train the model from scratch on the dataset, you can run the provided training script. Make sure the dataset logic handles the MNIST digit collection setup properly.

```bash
python training/Air_writing_detector_training.py
```

It creates a synthetic math evaluation set and scales using augmentation. The generated `scientific_model1.h5` receives automatic routing into the `model/` folder.

---

## Next Steps

- Integrate with **MediaPipe** hand landmark tracking.
- Build the real-time canvas / stroke accumulation layer.
- Ensure algebraic support with advanced parser implementation.
