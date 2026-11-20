# Teder

### AI-Powered Music Genre Classification & Personalized Recommendations

**Teder** (originally developed as **BeatSense**) is an end-to-end machine learning application for automated music genre classification and personalized music recommendations.

The system combines **Digital Signal Processing (DSP)**, **Convolutional Neural Networks (CNNs)**, and user-defined preferences to analyze audio tracks, identify their primary musical genres, and generate personalized **"Like"** or **"Dislike"** recommendations.

By extracting Mel-Frequency Cepstral Coefficients (MFCCs) from raw audio, Teder transforms audio signals into structured representations suitable for deep learning.

---

## 01. Key Features

### Audio Processing & Feature Extraction

- **MFCC Extraction:** Converts `.wav` and `.mp3` audio files into standardized `40 x 200` feature matrices.
- **Audio Normalization:** Applies a shared feature scaler to ensure consistent model inputs.
- **Sliding-Window Analysis:** Processes multiple segments of an audio track for more comprehensive predictions.
- **Multi-Format Support:** Accepts local audio files and YouTube URLs.

### Deep Learning Architecture

Built using **TensorFlow** and **Keras**, the classification network incorporates:

- Multiple `Conv2D` layers for hierarchical feature extraction.
- Batch normalization for training stability.
- Spatial dropout to reduce overfitting.
- L2 regularization for improved generalization.
- LeakyReLU activation functions for nonlinear feature learning.

### Advanced Data Augmentation

To improve robustness and generalization when training with limited datasets, Teder implements:

| Technique             | Purpose                                                       |
| --------------------- | ------------------------------------------------------------- |
| Time Stretching       | Introduces tempo variations                                   |
| Noise Injection       | Improves resistance to background noise                       |
| Frequency Masking     | Simulates missing frequency information                       |
| Time Masking          | Improves temporal robustness                                  |
| Composite Crossfading | Creates additional training variations through audio blending |

### Interactive Gradio Dashboard

Teder provides an integrated web interface built with **Gradio**, organized into five functional modules.

| Module            | Description                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| **Train Model**   | Start training sessions, monitor progress through custom callbacks, and visualize training metrics. |
| **Model Arena**   | Compare two saved models using uploaded audio tracks or YouTube videos.                             |
| **Predict**       | Perform multi-segment genre classification on audio files or YouTube URLs.                          |
| **Settings**      | Manage saved models, promote production models, inspect battle statistics, and clear logs.          |
| **Configuration** | Inspect neural network hyperparameters and DSP configuration settings.                              |

YouTube audio processing is supported through `yt-dlp`.

---

## 02. System Architecture

### Audio Configuration

| Parameter              | Value          |
| ---------------------- | -------------- |
| Sample Rate            | `22,050 Hz`    |
| Audio Segment Duration | `4 seconds`    |
| Samples per Segment    | `88,200`       |
| MFCC Coefficients      | `40`           |
| MFCC Time Frames       | `200`          |
| Model Input Shape      | `(40, 200, 1)` |

### Training Configuration

| Parameter               | Value                           |
| ----------------------- | ------------------------------- |
| Framework               | TensorFlow / Keras              |
| Optimizer               | Adam                            |
| Loss Function           | Sparse Categorical Crossentropy |
| Batch Size              | `24`                            |
| Learning Rate           | `0.0005`                        |
| Label Smoothing         | `0.05`                          |
| L2 Regularization       | `0.0032`                        |
| Spatial Dropout         | `0.27 - 0.46`                   |
| Early Stopping Patience | `6 epochs`                      |
| Early Stopping Monitor  | Validation Loss                 |

All configurable values are centralized in constants to maintain consistency, improve maintainability, and eliminate hardcoded configuration values throughout the application.

---

## 03. Project Structure

The application automatically organizes datasets, trained models, evaluation results, and production artifacts within the working directory or a connected Google Drive environment.

**Default Google Drive directory:** `final-project:BeatSense`

```
Teder/
│
├── data/
│   └── processed_data/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models_history/
│   ├── *.keras
│   └── *.pkl
│
├── metrics/
│   ├── *.png
│   └── *.txt
│
├── best_model.keras
├── mfcc_scaler.pkl
└── README.md
```

### Directory Overview

| Path                   | Description                                                         |
| ---------------------- | ------------------------------------------------------------------- |
| `data/processed_data/` | Genre-organized training, validation, and testing datasets.         |
| `models_history/`      | Timestamped model checkpoints and associated scaler configurations. |
| `metrics/`             | Generated confusion matrices and classification reports.            |
| `best_model.keras`     | Currently active production model.                                  |
| `mfcc_scaler.pkl`      | Normalization scaler associated with the active model.              |
| `README.md`            | Project documentation.                                              |

---

## 04. Installation & Setup

### Prerequisites

- Python 3.10 or newer.
- pip package manager.
- Required machine learning and audio processing dependencies.

### Install Dependencies

```
pip install numpy matplotlib librosa tensorflow gradio yt-dlp scikit-learn
```

### Launch the Application

Run the notebook cells or execute the application script containing the Gradio interface.

To launch the interactive dashboard:

```
demo.launch(share=True)
```

This starts the Gradio application and enables access through a shareable interface.

### Automated Training Mode

To enable the configured automatic training execution path, set the environment variable:

```
export BEATSENSE_AUTORUN_MAIN_TRAINING="1"
```

This mode allows training to be initiated without manually using the Gradio dashboard.

---

## 05. Model Evaluation

Every completed training cycle evaluates the model using the weights restored from the checkpoint with the lowest validation loss.

Evaluation artifacts are automatically exported to the `metrics/` directory.

### Classification Report

Provides detailed per-genre classification metrics:

- **Precision:** Proportion of predictions for a genre that are correct.
- **Recall:** Proportion of actual genre samples correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **Support:** Number of evaluation samples belonging to each genre.

### Confusion Matrix

Generates a visual representation of predicted versus actual genre labels.

The confusion matrix helps identify:

- Genres that are consistently classified correctly.
- Genre pairs that are frequently confused.
- Class-specific prediction weaknesses.
- Potential areas for dataset or model improvement.

---

## 06. Model Management & Comparison

### Model History

Trained models are preserved as timestamped `.keras` artifacts alongside their corresponding scaler configurations.

This enables experimentation with different training sessions without overwriting previously saved models.

### Model Arena

The Model Arena supports direct comparisons between two saved models using the same audio input.

**Supported inputs:**

- Uploaded audio files.
- YouTube video URLs.

The comparison workflow enables users to examine differences in model predictions and evaluate competing model versions.

### Production Model Promotion

A selected model can be promoted through the Settings interface.

The active production artifacts are:

```
best_model.keras
mfcc_scaler.pkl
```

Maintaining the correct model-scaler pairing ensures that inference uses the same feature normalization configuration as training.

---

## 07. End-to-End Processing Pipeline

The complete Teder pipeline consists of the following stages:

```
              Audio Input
                   |
                   v
           Audio Preprocessing
                   |
                   v
            MFCC Extraction
                   |
                   v
          Feature Normalization
                   |
                   v
           CNN Classification
                   |
                   v
          Genre Probabilities
                   |
                   v
          Primary Genre Prediction
                   |
                   v
         User Preference Matching
                   |
                   v
          Like / Dislike Result
```

The prediction system combines genre classification results with configured user preferences to generate personalized recommendations.

---

## 08. Technology Stack

| Category                   | Technologies                      |
| -------------------------- | --------------------------------- |
| Programming Language       | Python                            |
| Deep Learning              | TensorFlow, Keras                 |
| Audio Processing           | Librosa                           |
| Numerical Computing        | NumPy                             |
| Machine Learning Utilities | Scikit-learn                      |
| Data Visualization         | Matplotlib                        |
| Web Interface              | Gradio                            |
| YouTube Integration        | yt-dlp                            |
| Model Serialization        | Keras (`.keras`), Pickle (`.pkl`) |
| Storage                    | Local Filesystem, Google Drive    |

---

## 09. Project Objective

Teder demonstrates the integration of **audio signal processing**, **deep learning**, and **personalized recommendation logic** within a unified machine learning application.

The project focuses on four primary objectives:

1. **Automated Genre Recognition:** Classify music tracks using learned audio representations.
2. **Reliable Model Training:** Apply augmentation, normalization, and regularization techniques to improve classification robustness.
3. **Model Experimentation:** Support repeatable training, performance evaluation, and direct model comparisons.
4. **Personalized Music Discovery:** Translate predicted genres into preference-based music recommendations.

---

### Teder - AI Music Classification & Recommendation System

**Built with Python, TensorFlow, Librosa, and Gradio.**

Add a clear usage exampleDocument expected input formats
