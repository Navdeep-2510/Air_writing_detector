import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

# --- CONFIGURATION ---
NUM_SAMPLES = 6000 
IMG_SIZE = 28

# --- THE LEAN ALGEBRA MAPPING (22 Classes) ---
SYMBOL_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '+', 11: '-', 12: '*', 13: '/',
    14: '=', 15: '.', 16: '(', 17: ')',
    18: '^', 19: 'v', 20: 'x', 21: 'y'
}

def generate_symbol(symbol, num_samples):
    images, labels = [], []
    label_id = [k for k, v in SYMBOL_MAPPING.items() if v == symbol][0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for _ in range(num_samples):
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        
        scale = np.random.uniform(0.7, 1.1)
        thickness = np.random.randint(2, 4) 
        angle = np.random.randint(-15, 15)
        shift_x = np.random.randint(-3, 3)
        shift_y = np.random.randint(-3, 3)

        draw_symbol = symbol
        
        # Geometric Powers and Roots
        if symbol == '^':
            pt1 = (5 + np.random.randint(-2,2), 22)
            pt2 = (14, 6 + np.random.randint(-2,2))
            pt3 = (23 + np.random.randint(-2,2), 22)
            cv2.line(img, pt1, pt2, (255), thickness)
            cv2.line(img, pt2, pt3, (255), thickness)
            draw_symbol = None 

        elif symbol == 'v':
            pt1 = (5, 6)
            pt2 = (14, 22)
            pt3 = (23, 6)
            cv2.line(img, pt1, pt2, (255), thickness)
            cv2.line(img, pt2, pt3, (255), thickness)
            draw_symbol = None

        if draw_symbol:
            text_size = cv2.getTextSize(draw_symbol, font, scale, thickness)[0]
            text_x = (IMG_SIZE - text_size[0]) // 2 + shift_x
            text_y = (IMG_SIZE + text_size[1]) // 2 + shift_y
            cv2.putText(img, draw_symbol, (text_x, text_y), font, scale, (255), thickness)

        M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1)
        img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))

        img = img.astype('float32') / 255.0
        images.append(img.reshape(28, 28, 1))
        labels.append(label_id)
        
    return np.array(images), np.array(labels)

print("--- 1. Loading Digits (MNIST) ---")
(raw_img, raw_lbl), _ = tf.keras.datasets.mnist.load_data()
raw_img = raw_img.reshape(-1, 28, 28, 1).astype('float32') / 255.0

balanced_mnist_imgs, balanced_mnist_lbls = [], []
for i in range(10):
    idx = np.where(raw_lbl == i)[0]
    sampled_indices = np.random.choice(len(idx), NUM_SAMPLES, replace=(len(idx) < NUM_SAMPLES))
    balanced_mnist_imgs.append(raw_img[idx][sampled_indices])
    balanced_mnist_lbls.append(raw_lbl[idx][sampled_indices])

t_img = np.vstack(balanced_mnist_imgs)
t_lbl = np.concatenate(balanced_mnist_lbls)

print(f"--- 2. Generating Core Algebra Symbols ({NUM_SAMPLES} each) ---")
sym_imgs, sym_lbls = [], []
targets = ['+', '-', '*', '/', '=', '.', '(', ')', '^', 'v', 'x', 'y']

for char in targets:
    print(f"   Generating '{char}'...")
    imgs, lbls = generate_symbol(char, NUM_SAMPLES)
    sym_imgs.append(imgs)
    sym_lbls.append(lbls)

final_img = np.vstack([t_img, np.vstack(sym_imgs)])
final_lbl = np.concatenate([t_lbl, np.concatenate(sym_lbls)])

indices = np.arange(len(final_lbl))
np.random.shuffle(indices)
final_img, final_lbl = final_img[indices], final_lbl[indices]

print(f"--- 3. Training on {len(final_lbl)} images ---")
# Shrunk the output layer to 22 Classes
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(22, activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(final_img, final_lbl, epochs=6, batch_size=128)

model.save("model/scientific_model1.h5")
print("✅ Success! Lean Algebra Brain created.")