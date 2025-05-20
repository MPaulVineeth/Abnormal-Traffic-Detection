# 🚦 Abnormal Traffic Detection using ABS-CNN with CBAM

This final year project implements an advanced deep learning model to detect abnormal traffic in network data. It uses a custom **ABS-CNN (Attention-based CNN)** architecture enhanced with the **CBAM (Convolutional Block Attention Module)** to analyze .pcap files converted into 2D image format.

## 📌 Objectives
- 🧠 Build a custom CNN with attention to detect abnormal traffic
- 🔄 Convert .pcap files into 28×28 image arrays
- 📊 Train and evaluate using real packet data
- 🎯 Achieve high accuracy in abnormal traffic classification

## 🏗️ Model Architecture: ABS-CNN with CBAM
- **3 Convolution Layers**: extract features from image data
- **CBAM Attention Modules**: apply both channel and spatial attention
- **Dense Layers**: classify into known traffic types
- **Input Shape**: (28, 28, 2) images from .pcap packet bytes

## 🗂️ Project Structure
```
abnormal-traffic-detection/
├── data_preprocessing/
│   ├── preprocess.py        # Convert .pcap to image
│   └── dataset/images/      # Saved .npy images
├── pcaps/                   # Raw .pcap files (Skype.pcap, etc.)
├── model/
│   ├── abs_cnn.py           # ABS-CNN model architecture
│   └── cbam.py              # CBAM attention module
├── train.py                 # Train the ABS-CNN model
├── evaluate.py              # Evaluation: confusion matrix, F1
├── plot_history.py          # Plot training accuracy/loss
├── abs_cnn_model.h5         # Saved trained model
├── history.pkl              # Training history
├── training_plot.png        # Accuracy & loss curves
├── confusion_matrix.png     # Class-wise performance
├── requirements.txt
└── README.md
```

## ⚙️ How to Run the Project
> 💡 Ensure Python 3.7+ is installed and your virtual environment is activated.

### 1. 🛠️ Install Requirements
```
pip install -r requirements.txt
```

### 2. 📥 Preprocess `.pcap` files
```
python data_preprocessing/preprocess.py
```
This converts each `.pcap` into a 28×28 grayscale image saved as `.npy`.

### 3. 🧠 Train the Model
```
python train.py
```
This will:
* Train ABS-CNN on `.npy` image data
* Save `abs_cnn_model.h5` and `history.pkl`

### 4. 🧪 Evaluate the Model
```
python evaluate.py
```
This prints:
* Classification report (Precision, Recall, F1-score)
* Saves: `confusion_matrix.png`

### 5. 📈 Visualize Training Performance
```bash
python plot_history.py
```
This saves: `training_plot.png`

## 🎯 Final Results

| Metric | Value |
|--------|-------|
| ✅ Accuracy | 100.00% |
| 📉 Loss | 0.0000 |
| 📊 F1-Score | 1.00 |

* Model perfectly detects abnormal traffic types such as:
   * Skype
   * Facetime
   * BitTorrent
   * Zeus (malware)
   * Tinba, etc.

## 📊 Visual Outputs
* `confusion_matrix.png` – visualizes class-wise prediction accuracy
* `training_plot.png` – accuracy/loss vs. epoch

## 📚 Research Basis
Paper: **"Abnormal Traffic Detection Based on Attention and Big Step Convolution"**
Implemented using TensorFlow, Scapy, OpenCV, and Python.

## 👨‍💻 Developed By
**M Paul Vineeth**
BTech Final Year Project – Vellore Institute of Technology, VIT Vellore.
Guide: Dr. Meenakshi S P.

## 💡 Final Note
This project demonstrates how attention-based deep learning can enhance cybersecurity tools, specifically in real-time abnormal traffic classification.

🎓 Designed for academic review, real-world relevance, and research excellence.
