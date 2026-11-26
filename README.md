# cat-vs-dog-classifier
Cat vs Dog image classifier using CNN and transfer learning (TensorFlow/Keras)
# 🐱🐶 Cat vs Dog Image Classifier

A deep learning project that tells cats from dogs in photos.
I improved the model step by step: first a CNN built from scratch,
then fixing overfitting, then transfer learning with MobileNetV2.

![Demo](images/demo_screenshot.png)

## Results

| Step | Approach | Validation accuracy |
|------|----------|---------------------|
| 1 | CNN from scratch | ~65% (overfitting) |
| 2 | + Data augmentation + Dropout | ~74% |
| 3 | Transfer learning (MobileNetV2) | ~97% |

### Learning curves (transfer learning)
![Curves](images/stage4_curves.png)

### Confusion matrix
![Confusion matrix](images/confusion_matrix.png)

## What I did, step by step
1. **Loaded the data**: 2,000 training and 1,000 validation photos
   (cats vs dogs, balanced classes).
2. **Built a CNN from scratch**: it memorized the training photos
   (97% train accuracy but only 65% validation accuracy).
3. **Fixed overfitting**: added data augmentation (flip, rotate, zoom)
   and dropout.
4. **Used transfer learning**: reused MobileNetV2, which was pretrained on
   ImageNet, and trained only a small new head.
5. **Evaluated**: confusion matrix and a look at the wrong predictions.
6. **Built a demo**: a Gradio web app where you upload a photo.

## What I learned
- How convolution, filters and pooling work in a CNN
- How to spot overfitting from learning curves
- Why data augmentation and dropout help
- Why transfer learning works well with small datasets
- Why accuracy alone is not enough (confusion matrix)

## Limitations
- The model only knows two classes. For a photo with **both** animals,
  or **neither** (like a car), it still gives a cat/dog answer,
  which can be misleading.
- It was trained on only 2,000 photos, so unusual photos
  (very blurry, tiny animal) can fail.
- Photos with probability between 35% and 65% are shown as "Not sure".

## Run it yourself
```bash
git clone https://github.com/YOUR-USERNAME/cat-vs-dog-classifier.git
cd cat-vs-dog-classifier
pip install -r requirements.txt
python app.py
```
Open the local link shown in the terminal and upload a photo.

## Dataset
[microsoft/cats_vs_dogs](https://huggingface.co/datasets/microsoft/cats_vs_dogs)
on Hugging Face. The dataset is not included in this repo.

## Tech stack
Python · TensorFlow · Keras · MobileNetV2 · Gradio · NumPy · Matplotlib

## Future improvements
- Grad-CAM heatmaps to show where the model looks
- Multi-label detection (cat and dog in the same photo)
- Permanent live demo on Hugging Face Spaces
