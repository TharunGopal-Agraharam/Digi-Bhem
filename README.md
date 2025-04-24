# CIFAR Image Classification

This project focuses on image classification using the CIFAR dataset, employing deep learning models to recognize objects in images. The work demonstrates model training, evaluation, and visualization of predictions using popular libraries such as TensorFlow/Keras or PyTorch.

## Dataset

The CIFAR dataset is a collection of images commonly used for training machine learning and computer vision algorithms. Each image is 32x32 pixels in size and falls into one of the 10 categories.

## Features

- Data loading and preprocessing
- Deep learning model implementation (e.g., CNN)
- Model training and evaluation
- Visualization of predictions
- Model accuracy and performance analysis

## Technologies Used

- Python
- Jupyter Notebook
- TensorFlow/Keras or PyTorch
- Matplotlib
- NumPy

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/cifar-image-classification.git
   cd cifar-image-classification

2. Install the dependencies:

pip install -r requirements.txt


3. Run the notebook: Open CIFAR.ipynb in Jupyter Notebook or JupyterLab and run the cells sequentially.


# Model Summary 

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 32, 32, 64)     │           256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 30, 30, 64)     │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 28, 28, 64)     │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 14, 14, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 12, 12, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 10, 10, 128)    │       147,584 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 5, 5, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 5, 5, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 3200)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 128)            │       409,728 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 10)             │         1,290 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 706,572 (2.70 MB)
 Trainable params: 706,570 (2.70 MB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 2 (12.00 B)

# Results
  - Model trained on CIFAR dataset
  - Achieved high accuracy
