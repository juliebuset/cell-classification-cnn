# White Blood Cell Image Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview
This project automates the **classification of white blood cells (leukocytes)** using machine learning and deep learning techniques. White blood cells are essential for the immune system, and accurate classification is crucial for diagnosing infections, inflammatory diseases, and certain cancers.

Manual classification is often **time-consuming and error-prone**, motivating automated approaches. This project compares three models for classifying white blood cell images: **Random Forest**, a **custom CNN**, and a **fine-tuned VGG16**.

> **Dataset:** Implemented on **Kaggle** using [Blood Cells Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) with **12,500 labeled images**.

---

## Dataset
The dataset contains images labeled into four categories:

| Cell Type     | Description |
|---------------|-------------|
| **Lymphocyte** | Small, dark round nucleus, minimal cytoplasm. Adaptive immunity. |
| **Monocyte**   | Large, kidney-shaped nucleus, transforms into macrophages. |
| **Eosinophil** | Bilobed nucleus with reddish granules, parasite/allergy defense. |
| **Neutrophil** | Multilobed nucleus, granular cytoplasm, first line of defense. |

> Note: Monocytes and neutrophils are morphologically similar, increasing classification difficulty.

---

## Project Structure

1. **Data Preprocessing & Augmentation**
   - Images resized to **128x128 pixels**
   - **Edge detection** and **bounding box extraction** to isolate cells
   - Processed images saved in structured folders for training

2. **Data Splitting**
   - 80% training / 20% validation
   - Labels one-hot encoded for model compatibility

3. **Models**
   - **Random Forest:** baseline model, trained on flattened images
   - **VGG16 Fine-tuned:** pre-trained on ImageNet, frozen base layers, custom FC layers
   - **Custom CNN:** tailored convolutional architecture for optimal feature extraction

4. **Evaluation**
   - Metrics: **Accuracy, Precision, Recall, F1-score**
   - Confusion matrices and training/validation curves for comparison

---

## Results

| Model           | Accuracy | Precision | Recall | F1-score |
|-----------------|---------|-----------|--------|----------|
| Random Forest   | 79%     | 97%       | 79%    | 87%      |
| VGG16 Fine-tuned| 91%     | 91%       | 91%    | 91%      |
| Custom CNN      | 96%     | 96%       | 96%    | 96%      |

**Insights:**
- The **Custom CNN** dominates across all metrics, achieving high accuracy and balanced performance.
- Random Forest shows high precision but lower recall, especially for eosinophils and neutrophils.
- Fine-tuned VGG16 performs well but slightly below the custom CNN.

---

## Visual Preview

![Sample White Blood Cells](https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif)  
*Example of preprocessed white blood cell images.*

---

## Future Improvements
- **Ensemble Learning:** combine RF, VGG16, and CNN predictions
- **Advanced fine-tuning:** deeper VGG16 layers with lower learning rates
- **Resampling/augmentation:** improve minority class representation

---

## Usage

1. Clone the repo:  
```bash
git clone <repo_url>
```

2. Run the notebook **`white_blood_cell_classification.ipynb`** to:

- Preprocess images
- Train and evaluate models
- Visualize results

---

## Dependencies

- Python 3.8+
- TensorFlow / Keras
- OpenCV
- scikit-learn
- matplotlib
- seaborn
- numpy

---

## License

This project is licensed under the [MIT License](LICENSE).

