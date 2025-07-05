## MNIST digit classifier neural network model from scratch

**Ah yes the same ol’ MNIST handwritten digit prediction model... but from scratch** 

This project implements a simple feedforward neural network   fully coded from scratch using only NumPy  to classify digits from the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/).  
No TensorFlow. No PyTorch (except for loading MNIST). Just matrix and math !!

---

### What It Does : 

- Loads and preprocesses the MNIST dataset (28×28 grayscale digits)
- Implements a 3-layer neural network:
     - Input → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
- Trains using categorical cross-entropy loss & backpropagation
- Evaluates test accuracy
- Predicts and visualizes any test image

---

### Project Structure : 

MNIST-NN-FROM-SCRATCH/  
│  
├── models/  
│ └── neural_network.py # Core neural network logic  
│  
├── utils/  
│ ├── dataset_loader.py # Loads and preprocesses MNIST  
│ ├── metrics.py # Accuracy function  
│ └── visualization.py # Plot single predictions  
│  
├── notebooks/  
│ └── model.ipynb # Notebook for exploration  
│  
├── main.py # Entry point for training and prediction  
├── requirements.txt # Python dependencies  
└── .gitignore # Ignores **pycache**, etc.


---

### Installation :

1. **Clone the repo:**

```bash
git clone https://github.com/your-username/mnist-nn-from-scratch.git
cd mnist-nn-from-scratch
```

2. **Create virtual environment:**
   
``` bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
 pip install -r requirements.txt
```

### Running the model

1. Run the main file
   
```bash
python main.py
```

