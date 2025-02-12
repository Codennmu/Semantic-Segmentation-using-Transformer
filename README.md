# Semantic-Segmentation-using-Transformer
This repository contains the implementation of SegFormer-based segmentation for baggage threat detection using the Sixray dataset. The project fine-tunes Hugging Face's pretrained SegFormer model to achieve high accuracy in semantic segmentation. The trained model will serve as the teacher model for future knowledge distillation and incremental learning, forming a crucial part of the master's thesis:

**"Transformer and Knowledge Distillation-Based Baggage Threat Detection."**

## Features
- **Pretrained SegFormer model** from Hugging Face
- **Dataset preprocessing** for segmentation
- **Fine-tuning on Sixray dataset** for optimal accuracy
- **Grad-CAM visualization** to interpret model predictions
- **t-SNE visualization** for feature distribution analysis
- **Future use:** Teacher model for **incremental learning**

## Installation
Ensure you have Python 3.8+ and install the required dependencies:

```bash
pip install -U git+https://github.com/qubvel/segmentation_models.pytorch
pip install pytorch-lightning transformers datasets roboflow
pip install accelerate evaluate segmentation-models
```

## Dataset
The **Sixray dataset** contains X-ray images of baggage with concealed threats. The dataset includes:
- **Threat classes**: Knife, Gun, and other prohibited items
- **Background images**: Baggage without threats

## Training Pipeline
1. **Load Pretrained SegFormer**
    ```python
    from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
    feature_extractor = SegformerFeatureExtractor.from_pretrained('nvidia/segformer-b2-finetuned-ade-512-512')
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b2-finetuned-ade-512-512')
    ```

2. **Dataset Preprocessing**
    ```python
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    ```

3. **Fine-Tuning on Sixray Dataset**
    ```python
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        logging_dir='./logs',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    ```

4. **Grad-CAM Visualization**
    ```python
    import torch
    from pytorch_grad_cam import GradCAM
    target_layers = [model.segformer.encoder.layer[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor.unsqueeze(0))
    ```

5. **t-SNE Analysis**
    ```python
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(feature_vectors)
    ```

## Results
- **Best Accuracy:** Achieved after fine-tuning SegFormer
  ![Picture1](https://github.com/user-attachments/assets/828b1501-4d5a-470c-9ba1-de2521bff9ee)

- **Grad-CAM** shows attention regions for threat detection
  ![gradcam](https://github.com/user-attachments/assets/ec7accb7-d4af-4d7c-9481-75e3d90804ae)

- **t-SNE visualization** differentiates feature clusters
  ![tsne visulization graph](https://github.com/user-attachments/assets/56a464ef-5bc2-454f-9130-657e8f38b460)


## Next Steps
The trained **teacher model** will be used for:
1. **Knowledge distillation** to train a lightweight student model
2. **Incremental learning** for real-world baggage screening

## Citation
If you use this work, please cite:
```
@article{your_citation,
  title={Transformer and Knowledge Distillation-Based Baggage Threat Detection},
  author={Saad Mazhar Khan},
  year={2025}
}
```

