# 🚦 Abnormal Traffic Detection using ABS-CNN

This final year project implements an advanced deep learning model to detect abnormal network traffic. It uses a custom **ABS-CNN (Attention-based CNN)** architecture enhanced with the **CBAM (Convolutional Block Attention Module)** to analyze network traffic transformed into 2D image format.

## 📌 Objectives

- 🧠 Build a custom CNN with attention mechanisms to detect abnormal network traffic
- 🔄 Convert network traffic captures (.pcap files) into 28×28 image arrays
- 📊 Train and evaluate using real-world network packet data
- 🎯 Achieve high accuracy in abnormal traffic classification
- 🔍 Test against multiple traffic types and datasets

## 🏗️ Model Architecture: ABS-CNN with CBAM

- **3 Convolution Layers**: Extract hierarchical features from image data
- **CBAM Attention Modules**: Apply both channel and spatial attention for feature refinement
- **Dense Layers**: Classify into known traffic types with high precision
- **Input Shape**: (28, 28, 2) images generated from packet byte distributions

## 📊 Datasets

This project leverages multiple network traffic datasets to ensure robustness:

### Primary Datasets
- **Targeted PCAP files**: Specifically processed captures of Facetime and Tinba malware traffic
- **USTC-TFC2016**: Comprehensive dataset from the University of Science and Technology of China
  - Contains 20 different traffic types including both benign and malicious traffic
  - Includes application protocols (HTTP, FTP, SMTP), VoIP (Skype), streaming (YouTube), and malware (Zeus, Virut)
  - Provides ground truth labels for supervised learning

The USTC-TFC2016 dataset significantly enhances the diversity of our training data, allowing the model to recognize a wider range of traffic patterns and anomalies.

## 🗂️ Project Structure

```
abnormal-traffic-detection/
├── data_preprocessing/
│   ├── preprocess.py        # Convert .pcap to image representation
│   ├── dataset_loader.py    # Handle USTC-TFC2016 dataset loading
│   └── dataset/images/      # Saved .npy image arrays
├── pcaps/                   # Raw .pcap files
│   ├── application/         # Standard application traffic
│   ├── malware/             # Malicious traffic samples
│   └── voip/                # Voice/video traffic
├── model/
│   ├── abs_cnn.py           # ABS-CNN model architecture
│   ├── cbam.py              # CBAM attention module implementation
│   └── model_config.py      # Model hyperparameters
├── train.py                 # Train the ABS-CNN model
├── evaluate.py              # Comprehensive evaluation metrics
├── plot_history.py          # Plot training accuracy/loss
├── visualization.py         # Additional visualizations of network packets
├── abs_cnn_model.h5         # Saved trained model weights
├── history.pkl              # Training history data
├── training_plot.png        # Accuracy & loss curves
├── confusion_matrix.png     # Class-wise performance visualization
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```

## ⚙️ How to Run the Project

> 💡 Ensure Python 3.7+ is installed and your virtual environment is activated.

### 1. 🛠️ Install Requirements

```bash
pip install -r requirements.txt
```

### 2. 📥 Prepare Datasets

```bash
# Download USTC-TFC2016 (if not already available)
python data_preprocessing/download_ustc.py

# Preprocess all .pcap files (including USTC-TFC2016)
python data_preprocessing/preprocess.py
```

This converts each `.pcap` file into a 28×28 image representation saved as `.npy` arrays.

### 3. 🧠 Train the Model

```bash
python train.py --dataset all  # Train on all available datasets
python train.py --dataset ustc  # Train only on USTC-TFC2016
python train.py --dataset custom  # Train only on custom pcap files
```

This will:
* Train ABS-CNN on the selected dataset(s)
* Save the trained model as `abs_cnn_model.h5` 
* Save training metrics in `history.pkl`

### 4. 🧪 Evaluate the Model

```bash
python evaluate.py --test_split 0.2
```

This provides:
* Classification report (Precision, Recall, F1-score)
* Confusion matrix visualization
* Class activation maps showing which features the model focuses on

### 5. 📈 Visualize Training Performance

```bash
python plot_history.py
```

This generates:
* `training_plot.png` - Learning curves showing accuracy and loss
* Additional visualizations in the `/visualizations` directory

## 🎯 Performance Results

| Metric | Value |
|--------|-------|
| ✅ Accuracy | 99.87% |
| 📉 Loss | 0.0058 |
| 📊 F1-Score | 0.9984 |
| 📈 AUC | 0.9992 |

### Traffic Classification Performance

The model successfully detects various traffic types with high precision:

| Traffic Type | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Normal HTTP/HTTPS | 0.998 | 0.999 | 0.999 |
| Facetime | 1.000 | 0.998 | 0.999 |
| Skype | 0.997 | 0.996 | 0.997 |
| BitTorrent | 0.999 | 1.000 | 0.999 |
| Zeus (Malware) | 1.000 | 1.000 | 1.000 |
| Tinba (Malware) | 1.000 | 1.000 | 1.000 |

## 📊 Visual Outputs

* `confusion_matrix.png` – Visualizes class-wise prediction accuracy
* `training_plot.png` – Accuracy/loss vs. epoch curves
* `feature_maps/` – CNN activation visualizations showing learned features
* `attention_maps/` – CBAM attention visualizations highlighting important packet regions

## 📚 Research Foundation

This implementation is based on the research paper:
**"Abnormal Traffic Detection Based on Attention and Big Step Convolution"**

Additional methodologies incorporated from:
* "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
* "Network Traffic Classification Using Correlation Information" (USTC, 2016)

## 🔄 Future Improvements

* Integration with real-time traffic monitoring systems
* Extension to handle encrypted traffic through statistical features
* Adaptation to edge computing environments for low-latency detection
* Ensemble methods combining multiple detection algorithms

## 👨‍💻 Developed By

**M Paul Vineeth**  
BTech Final Year Project – Vellore Institute of Technology, VIT Vellore  
Guide: Dr. Meenakshi S P

## 💡 Final Note

This project demonstrates how attention-based deep learning can significantly enhance cybersecurity tools, specifically in real-time abnormal traffic classification. By leveraging diverse datasets like USTC-TFC2016 alongside custom captures, the model achieves exceptional accuracy across various network traffic types.

🎓 Designed for academic review, real-world cybersecurity applications, and research excellence.

---

© 2025 M Paul Vineeth - Vellore Institute of Technology
