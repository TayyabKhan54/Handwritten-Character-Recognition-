Handwritten Character Recognition
===============================

Small project demonstrating handwritten character recognition using CNNs on MNIST / EMNIST.

Quick start
-----------

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Train on MNIST (downloads dataset automatically):

```bash
python train.py --dataset MNIST --epochs 5 --batch-size 64
```

3. Evaluate a saved checkpoint:

```bash
python evaluate.py --dataset MNIST --checkpoint checkpoint.pt
```

Notes
-----

- `EMNIST` support requires a torchvision build that includes `EMNIST` dataset. Use `--dataset EMNIST` and optionally `--emnist-split letters`.
