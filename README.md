# Question: 1

# Adversarial Robustness on MNIST

This project trains a simple Convolutional Neural Network (CNN) on the **MNIST** handwritten digit dataset and evaluates its robustness against adversarial attacks.

We implement and compare two adversarial methods:
- **FGSM (Fast Gradient Sign Method)**
- **Gaussian Noise Perturbation**

---

## Project Structure

```
├── test.py              # Main training, testing, and evaluation script
├── fgsm.py         # FGSM attack implementation
├── fgsm_gaussian.py     # Gaussian noise adversarial attack implementation
└── data/                # MNIST dataset (downloaded automatically)
```

---

## Getting Started

### 1. Install Dependencies
Make sure you have Python 3.8+ and install the required libraries:

```bash
pip install torch torchvision pandas numpy
```

### 2. Run Training and Evaluation

```bash
python test.py
```

This will:
1. Train the CNN on MNIST for 5 epochs  
2. Evaluate clean validation accuracy  
3. Run adversarial evaluation using FGSM and Gaussian noise  

---

## Model Architecture

The CNN (`SimpleNet`) has the following layers:
- **Conv2d(1 → 32, kernel=3)** + ReLU  
- **Conv2d(32 → 64, kernel=3)** + ReLU + MaxPool(2)  
- **Dropout (0.25)**  
- **Fully Connected (64×12×12 → 128)** + ReLU  
- **Dropout (0.5)**  
- **Fully Connected (128 → 10)** + Softmax  

---

## Initial Training & Testing Results

```
Epoch 1, Training Loss: 1.5542, Accuracy: 91.01%
Epoch 2, Training Loss: 1.5024, Accuracy: 95.94%
Epoch 3, Training Loss: 1.4963, Accuracy: 96.48%
Epoch 4, Training Loss: 1.4930, Accuracy: 96.81%
Epoch 5, Training Loss: 1.4928, Accuracy: 96.84%

Validation Loss: 1.4784, Accuracy: 98.27%
```

---

## Adversarial Robustness Results

### FGSM Attack

| Epsilon | Accuracy | Accuracy Drop |
|---------|----------|---------------|
| 0.010   | 97.58%   | 0.69%         |
| 0.050   | 97.47%   | 0.80%         |
| 0.100   | 97.26%   | 1.01%         |
| 0.200   | 96.77%   | 1.50%         |
| 0.250   | 96.47%   | 1.80%         |
| 0.300   | 96.25%   | 2.02%         |
| 0.500   | 92.26%   | 6.01%         |

---
### Gaussian Noise Attack

| Epsilon | Accuracy | Accuracy Drop |
|---------|----------|---------------|
| 0.010   | 97.61%   | 0.66%         |
| 0.050   | 97.62%   | 0.65%         |
| 0.100   | 97.60%   | 0.67%         |
| 0.200   | 97.57%   | 0.70%         |
| 0.250   | 97.58%   | 0.69%         |
| 0.300   | 97.50%   | 0.77%         |
| 0.500   | 96.74%   | 1.53%         |

---

## Key Observations

- The model achieves **~98% validation accuracy** on clean MNIST data.  
- FGSM attacks show a **larger accuracy drop** compared to Gaussian noise.  
- Small perturbations (`ε ≤ 0.1`) do not drastically affect accuracy.  
- Strong perturbations (`ε ≥ 0.3`) significantly reduce model performance.  

---

## Future Improvements
  
- Implement **batch-level adversarial evaluation** (instead of per-sample).  
- Compare with stronger attacks like **PGD (Projected Gradient Descent)**.  
- Visualize clean vs adversarial samples.  

---

---

## API Server (`app_fgsm.py`)

The FastAPI server provides endpoints to run **FGSM adversarial attacks** on MNIST digit images.

### Running the Server
```bash
python app_fgsm.py
```
- Server starts at: `http://localhost:3000`
- Interactive docs: [http://localhost:3000/docs](http://localhost:3000/docs)

### Endpoints

- **GET /health** → API health check  
- **GET /info** → metadata + endpoints  
- **POST /fgsm-attack** → JSON array input (28×28 pixels)  
- **POST /fgsm-upload** → Upload image file + run attack  
- **POST /batch-attack** → Run FGSM on multiple samples across epsilons  
- **GET /example** → Example usage + `curl` requests  

### Simple Model (For Testing)
- Input: 28×28 flattened → 784
- Hidden layers: 128 → 64
- Output: 10 classes
- Activations: ReLU + LogSoftmax

---

## API Client Tester (`test_api.py`)

Python script to **test all API endpoints**.

### Run Tests
```bash
python test_api.py
```

### Features
- Health check (`/health`)
- API metadata (`/example`)
- JSON input FGSM attack
- File upload FGSM attack (saves adversarial result)
- Batch attack across multiple epsilon values

### Example Output
```
=== Testing Batch Attack ===
✓ Batch attack successful
```

---

## Future Improvements
- Try stronger attacks (PGD, CW, DeepFool)
- Adversarial training for robustness
- Extend API for multiple datasets (e.g., CIFAR-10)
- Add visualization dashboard for adversarial examples

---

### Preview Outputs/Screenshots
Screenshots having outputs are in the folder named **output_pics**

---

## Author

Developed as part of an **Adversarial Machine Learning** experiment on MNIST.  
