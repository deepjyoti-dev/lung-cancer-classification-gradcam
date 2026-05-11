# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 14:50:36 2026
lung_dataset/
│
├── train/
│   ├── benign/
│   ├── malignant/
│   ├── normal/
│
├── test/
│   ├── benign/
│   ├── malignant/
│   ├── normal/

pip install tensorflow scikit-learn matplotlib seaborn opencv-python imbalanced-learn



🔥 Grad-CAM Visualization

Add this file after training.

🧠 GradCAM Code
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def show_gradcam(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    heatmap = make_gradcam_heatmap(img_array, model, "conv5_block3_out")

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.show()


# TEST GRADCAM
show_gradcam("lung_dataset/test/malignant/sample1.jpg")
@author: deepj
"""

import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


# ======================
# CONFIG
# ======================
IMG_SIZE = 224
BATCH = 16
EPOCHS = 15

train_dir = "lung_dataset/train"
test_dir = "lung_dataset/test"


# ======================
# DATA GENERATORS
# ======================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False
)


# ======================
# CLASS WEIGHTS (BALANCING)
# ======================
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(train_data.classes)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_data.classes
)

class_weights = dict(zip(classes, weights))
print("Class Weights:", class_weights)


# ======================
# TRANSFER MODEL
# ======================
base = ResNet50(weights='imagenet', include_top=False,
                input_shape=(IMG_SIZE, IMG_SIZE, 3))

base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ======================
# TRAIN
# ======================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights
)


# ======================
# EVALUATION
# ======================
y_probs = model.predict(test_data)
y_pred = np.argmax(y_probs, axis=1)
y_true = test_data.classes


# ======================
# METRICS
# ======================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

f1 = f1_score(y_true, y_pred, average='weighted')
print("F1 Score:", f1)


# ======================
# CONFUSION MATRIX
# ======================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=test_data.class_indices,
            yticklabels=test_data.class_indices)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ======================
# ROC (MULTI CLASS)
# ======================
from sklearn.preprocessing import label_binarize

y_bin = label_binarize(y_true, classes=range(train_data.num_classes))

plt.figure()
for i in range(train_data.num_classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} AUC={roc_auc:.2f}')

plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("Multi-Class ROC Curve")
plt.show()


# ======================
# SAVE MODEL
# ======================
model.save("lung_transfer_model.h5")
print("Model Saved")
