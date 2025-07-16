# PROJECT 2: Neural Network-Based Anomaly Detection in Sensor Data

This project includes two Python scripts designed to train and deploy a neural network for **real-time anomaly detection** in sensor data streams. The system is adaptable to various **time-series** and **IoT anomaly detection** use cases.

---

## Key Components

### Neural Network Training (`rec_project.py`)

- Built a custom **feedforward neural network** using **PyTorch**
- Trained on **normal sensor data** to learn baseline behavioral patterns
- Model architecture:
  - 3 fully connected layers
  - ReLU activation functions
  - Final Sigmoid output layer
- Training details:
  - Loss function: **Binary Cross-Entropy**
  - Optimizer: **Adam**
- The trained model is saved for later use in anomaly detection

### Real-Time Anomaly Detection (`detector.py`)

- Loads the previously trained neural network model
- Accepts **streaming or batch sensor data** for inference
- Preprocessing:
  - Converts input into PyTorch tensors
  - Normalizes input according to the training data scale
- Detection logic:
  - Applies thresholding on model predictions to classify anomalies
  - Logs or flags anomalous data points for alerting or analysis

### Implementation Details

- Core deep learning framework: **PyTorch**
- Modular script design for separating training and inference logic
- The model is general-purpose and can be adapted to:
  - Other types of sensor inputs
  - Different time-series anomaly detection tasks
  - Custom detection thresholds or alerting pipelines

---

## Summary

This end-to-end pipeline demonstrates how **neural networks** can be trained and deployed to detect anomalies in real-time data. The project emphasizes **modularity, adaptability**, and **scalability**, making it suitable for broader applications in the field of **machine learning for IoT and system monitoring**.
