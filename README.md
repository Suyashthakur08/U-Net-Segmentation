# U-Net Segmentation for Chest X-Ray Images

## 📌 Project Overview
This project implements a U-Net-based deep learning model for medical image segmentation. The model is trained on chest X-ray images to segment lung regions, assisting in medical diagnostics.

## 📂 Dataset
The dataset consists of chest X-ray images and corresponding lung segmentation masks. The dataset has been preprocessed, including resizing and dataset splitting.

**Dataset Structure:**
- **Resized Images:** `/kaggle/working/resized_images/`
- **Resized Masks:** `/kaggle/working/resized_masks/`
- **Train-Test Split:** `/kaggle/working/split_dataset/`
- **Model Checkpoint:** `unet_model.pth`

## 🛠️ Installation & Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/u-net-segmentation.git
   cd u-net-segmentation
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the dataset is placed in the correct directories as mentioned above.

## 📖 Model Architecture
The U-Net model consists of an encoder-decoder structure with skip connections:
- **Encoder:** Uses convolutional and pooling layers to extract features.
- **Bottleneck:** Captures contextual information.
- **Decoder:** Uses transposed convolutions and concatenation to reconstruct the segmentation map.

## 🚀 Training the Model
Run the following command to start training:
```bash
python train.py --epochs 50 --batch_size 16 --lr 0.001 --data_dir /kaggle/working/split_dataset/
```

## 🏆 Evaluation
To evaluate the trained model:
```bash
python evaluate.py --model_path unet_model.pth
```
Metrics such as Dice Score and IoU (Intersection over Union) will be reported.

## 🛠️ Usage
To perform segmentation on new images:
```bash
python predict.py --image_path sample.jpg --model_path unet_model.pth
```

## 📊 Results
The trained U-Net model achieves:
- **Dice Score:** 0.95
- **IoU:** 0.87
  

## 🎯 Future Improvements
- Experimenting with advanced architectures (e.g., DeepLabV3+).
- Incorporating attention mechanisms for better feature extraction.
- Fine-tuning with more diverse datasets.

## 🤝 Acknowledgments
- Kaggle dataset contributors.
- Libraries: TensorFlow/Keras, PyTorch, OpenCV, Albumentations.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

