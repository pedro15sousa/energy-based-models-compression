# Compressing EBMs

## Project Planning 
This section of the README details the planning and decision-making processes we underwent in our project. Our primary goal was to explore network compression techniques applied to energy-based models (EBMs) in an attempt to optimise performance and resource utilisation.

### Training MNIST classifier
- Objective: To create a classifier trained on the MNIST dataset, which was later used for computing the Frechet Inception Distance (FID) and Inception Score (IS).

### Experimenting with ResNet18 Variations
- Objective: To find a modified ResNet18 architecure to parameterise the energy function. 
- Approach: We experimented with several ResNet18 variants, analysing their compatibility and performance as energy functions. Some low performing architectures were reused as student models for knowledge distillation.
- Outcome: Different training attempts led us to the conclusion that the Swish activation function and an energy output computed as the average of the logits were the best design choices for stable training. 

### Challenges with Joint Energy Models (JEM)
- Objective: To train a JEM for simultaneous image generation and classification.
- Outcome: The training proved highly challenging and resource-intensive, with significant divergence observed. Consequently, we decided not to proceed with JEM for compression techniques.

### Quantization Efforts
- Objective: To apply quantization techniques to our models.
- Challenges: 
    - Incompatibility between the PyTorch quantization library and our custom Swish activation function. 
    - Necessity to customise the model class for quantizing and dequantizing tensors between layers.
- Outcome: We initially attempted quantization-aware training but eventually opted for post-training static quantization due to time and resource constraints.

### Exploring Pruning Techniques
- Objective: To apply both structured and unstructured pruning to our models.
- Approach:
    - We experimented with pruning on the global model and specifically on Conv2d layers.
    - For models with Conv2d layer-specific pruning, we conducted additional training for 7 epochs post-pruning.
- Limitations: Due to resource constraints, not all pruned models could be re-trained, especially given the extensive training time required for EBMs.

### Focus on Knowledge Distillation
- Objective: To implement knowledge distillation for efficient model training.
- Approach:
    - We developed simpler residual networks and straightforward CNNs to serve as student models.
    - These student models were trained using knowledge distilled from more complex networks. 
    - Experimented with different distillation losses, ended up choosing MSE because of the nature of scalar energy outputs of EBMs.

## Setup
As long as you have Python3 installed, run the following commands:
```python
python -m venv py_env
```
```bash
source py_env/bin/activate
```
```bash
pip install -r requirements.txt
```

## Getting started
The quickest way to get started with experimentation is to load some of the notebooks in Google Colab. Notebooks that end in analysis (e.g. `knowledge_distillation_analysis.ipynb`) are for experimenting with trained models and the ones ending in EBM (e.g. `knowledge_distillation_EBM.ipynb`) focus on training/generating the compressed version of the baseline, using the matching techniques. A Google Drive folder is currently available with the pre-trained baseline model, the structured and unstructured pruned models, and 6 different student models. Extra models include different architecture experimentaions for the baseline. Depending on the desired experimentation, the user should load the models locally in a subfolder in `saved_models/MNIST/` (e.g. place all pruned models in `saved_models/MNIST/pruned/`). Some notebooks might require adjusting the file paths depending on if it is being ran from Colab or locally.
Pre-trained models can be found in https://drive.google.com/drive/folders/1HJfPHmJdci1PkmCUN3Z6oLKN9o79VLrB?usp=sharing

Alternatively, the user can re-train the models, including the baseline. The energy functions can be found in `energy_funcs` and the user can experiment with multiple. It is important to note that the larger EBMs were trained on NVIDIA A100 GPUs for close to 3 days, which might be highly costly money-wise.




