# 📦 **Single-View 3D Object Reconstruction Using Deep Neural Networks**

Reconstructing a complete **3D voxel model** from a single 2D RGB image is one of the core challenges in computer vision.
This project implements a **deep neural network (CNN + 3D decoder)** that learns to map a single image into a 3D voxel grid, enabling the estimation of 3D object geometry from limited visual information.

This repository includes the full pipeline:

✔ Preprocessing and dataset conversion
✔ Neural network architecture
✔ Model training
✔ 3D voxel + mesh generation
✔ Visualization & inference scripts

---

## 🚀 **Project Demo**

| Input Image | 3D Reconstruction     |
| ----------- | --------------------- |
| 2D Image →  | Voxel / Mesh Output → |

*(Add your own sample outputs here after running inference.)*

---

## 🧠 **Model Architecture**

The model follows a **CNN Encoder → Fully Connected Bottleneck → 3D Decoder** design.

**Encoder (2D CNN):**

* Learns visual features from single-view RGB input
* Uses stacked convolution + batch norm + ReLU
* Outputs a compact latent embedding

**Latent Vector:**

* Dense representation capturing object structure

**Decoder (3D CNN):**

* Transposes convolutions to upsample
* Outputs a **32×32×32 voxel grid**
* Final activation: Sigmoid (for occupancy probability)

> 📌 Architecture diagram is included in repo:
> **`architecture_diagram.png`**

---

## 📂 **Repository Structure**

```
.
├── scripts/
│   └── pix3d_to_voxel_dataset.py        # Convert Pix3D to images + voxel grids
│
├── src/
│   ├── dataset.py                        # Dataset loader
│   ├── preprocess.py                     # Image preprocessing
│   ├── model.py                          # Encoder-decoder neural network
│   ├── train.py                          # Training pipeline
│   ├── utils.py                          # IoU, voxel slicing, mesh export
│   └── predict.py                        # Inference script
│
├── main.py                               # Train launcher
├── view_3d_model.py                      # Display .ply meshes (Trimesh)
├── requirements.txt                      # Dependencies
├── architecture_diagram.png              # Model architecture
├── 3D_Reconstruction_Project_Documentation_Upgraded.docx
└── README.md
```

---

## 📦 **Installation**

### 1️⃣ Clone repository

```
git clone https://github.com/<your-username>/Single-View-3D-Object-Reconstruction-Using-Deep-Neural-Networks.git
cd Single-View-3D-Object-Reconstruction-Using-Deep-Neural-Networks
```

### 2️⃣ Create virtual environment (optional)

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## 📁 **Dataset**

This project expects images + voxel grids.

You may use:

### **Pix3D Dataset**

* Not included in this repository (too large)
* Download from: [https://github.com/xingyuankuang/pix3d](https://github.com/xingyuankuang/pix3d)

Then convert to voxel format using:

```
python scripts/pix3d_to_voxel_dataset.py
```

This will produce:

```
dataset/
 ├── images/
 └── voxels/
```

---

## 🏋️ **Training the Model**

Run:

```
python main.py
```

Training includes:

* BCE loss on voxel occupancy
* IoU metric
* Model checkpoints
* Visualization of voxel slices per epoch

Results saved under:

```
results/saved_models/
results/visualizations/
```

---

## 🔍 **Inference (Reconstruction)**

To generate a 3D model from a single 2D image:

```
python -m src.predict --image path/to/image.jpg --model results/saved_models/reconstructor_epochXX.pth
```

Outputs:

* `pred_slices.png` → voxel slices
* `pred_mesh.ply` → 3D mesh (viewable in MeshLab / Blender)

Use this to visualize:

```
python view_3d_model.py
```

---

## 📈 **Evaluation**

Metrics:

* **IoU (Intersection-over-Union)** for voxel prediction
* Loss curves visualized per epoch
* Optionally export meshes for qualitative analysis

---

## 🧾 **.gitignore (Important)**

Large items **NOT uploaded** to repo:

* `dataset/`
* `pix3d/`
* `results/`
* `venv/`
* `*.ply`

(Already included in your repo.)

---

## 📚 **References**

1. **Pix3D Dataset** — Sun et al. *"Pix3D: Dataset and Methods for Single-View 3D Reconstruction"*, CVPR 2018
2. **3D CNNs** — Wu et al., *"Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling"*
3. PyTorch Documentation — [https://pytorch.org](https://pytorch.org)
4. Trimesh Library — [https://trimsh.org](https://trimsh.org)
5. Kingma & Ba — *Adam Optimizer*



