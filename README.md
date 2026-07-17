# SynapseVision

A Flask app that classifies brain MRI scans into 4 categories — glioma,
meningioma, pituitary tumor, or no tumor — using a fine-tuned VGG16 CNN.
Optionally generates a plain-language explanation of the result using a
vision LLM (Fireworks AI).

**Not a medical device.** Educational project only — not for clinical use.

## How it works

1. Upload an MRI scan through the web UI.
2. A VGG16 model (transfer learning, fine-tuned on MRI data) predicts one
   of the 4 classes.
3. (Optional) The scan + prediction are sent to Fireworks AI's vision
   model to generate a short plain-English explanation.

## Model performance

Trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
(7,023 images), tested on 1,600 held-out images.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Glioma | 0.844 | 0.665 | 0.744 |
| Meningioma | 0.773 | 0.833 | 0.801 |
| No Tumor | 0.858 | 0.995 | 0.921 |
| Pituitary | 0.931 | 0.908 | 0.919 |

Overall accuracy: **85.0%**. Full confusion matrix in
`models/evaluation_report.txt`.

## Setup

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd SynapseVision

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # add FIREWORKS_API_KEY if you want AI explanations
python main.py
```

Open `http://127.0.0.1:5000` and upload a scan. A trained model is
already included, so no training is required to try it out.

### Retraining
Dataset link above. Download it, unzip into `data/Training` and
`data/Testing` (class subfolders), then run `notebooks/brain_tumour.ipynb`.

## Tech stack
- Flask, TensorFlow/Keras
- VGG16 (transfer learning)
- Fireworks AI for the optional explanation text
- HTML/CSS/JS frontend, no framework
