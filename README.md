# cifar10-speedrun

This repository contains a script that trains a neural network to **94% accuracy on CIFAR-10 in just 2.095 seconds** on a single NVIDIA A100 GPU. This new record beats the previous best of 2.59 seconds.

| Script | Mean accuracy | Time |
| :--- | :--- | :--- |
| **`cifar10_speedrun.py`** | **94.01%** | **2.095s** |
| `airbench94_muon.py` | 94.01% | 2.59s |
| Standard ResNet-18 training | 96.0% | 7min |

\<br\>

## 🚀 Quickstart

To run the speedrun, simply clone the repository and run the Python script.

```bash
git clone https://github.com/your-username/cifar10-speedrun.git
cd cifar10-speedrun
pip install -r requirements.txt
python cifar10_speedrun.py
```

This will download the CIFAR-10 dataset and run the training. The script requires `torch` and `torchvision`.

\<br\>