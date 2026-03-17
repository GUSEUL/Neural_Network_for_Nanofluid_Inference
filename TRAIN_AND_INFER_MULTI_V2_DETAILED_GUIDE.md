# Technical Deep Dive: PhyCRNet Multi-Parameter Solver

This document provides a detailed explanation of the data flow, model I/O, and the dual-phase loss strategy used in `train_and_infer_multi_v2.py`.

---

## 1. Data Pipeline and Model I/O

The framework treats the fluid simulation as a spatiotemporal sequence prediction problem, conditioned by physical parameters.

### 1.1 Model Input
- **Spatiotemporal Sequence**: The model receives a sequence of previous states.
  - **Shape**: `[Batch, Sequence_Length, Channels, Height, Width]`
  - **Channels (4)**: $u$ (x-velocity), $v$ (y-velocity), $T$ (temperature), and $P$ (pressure).
  - **Default Sequence**: Usually 3 frames representing states at $t-2, t-1, t$.
- **HDF5 Caching (Performance Optimization)**: 
  - To accelerate training, `.mat` files are pre-processed into HDF5 format. 
  - This caches entire spatiotemporal sequences, reducing data loading time by 5-10x compared to reading `.mat` files on-the-fly.
- **Physical Parameters**: Four scalar values $(Ra, Ha, Q, Da)$ are provided.
  - These are normalized and injected into the network via **FiLM (Feature-wise Linear Modulation)** layers or concatenation to condition the flow field generation.

### 1.2 Model Output
- **Predicted State**: The model predicts the next state in the sequence.
  - **Shape**: `[Batch, Channels, Height, Width]`
  - **Target**: The flow field at $t+1$.

---

## 2. Training Phase: Loss Function

During training, the model weights are updated to learn the "Surrogate Mapping"—how the flow field evolves under specific physical parameters.

### 2.1 Training Loss Components
The total loss is a weighted sum:
$$\mathcal{L}_{train} = \mathcal{L}_{MSE} + \lambda (\mathcal{L}_{physics} + \mathcal{L}_{boundary})$$

1.  **Data Loss ($\mathcal{L}_{MSE}$)**: 
    - Standard Mean Squared Error between the predicted frame and the ground truth frame.
    - **Purpose**: Ensures the model captures the basic patterns of the fluid motion.
2.  **Physics Residual Loss ($\mathcal{L}_{physics}$)**:
    - Calculates the residuals of the Navier-Stokes and Energy equations using automatic differentiation or numerical gradients on the model output.
    - **Components**: Continuity, X-Momentum, Y-Momentum, and Energy residuals.
    - **Purpose**: Acts as a regularizer to ensure the predicted fields are physically plausible.
3.  **Boundary Loss ($\mathcal{L}_{boundary}$)**:
    - Penalizes deviations from known boundary conditions (e.g., zero velocity at walls, specific temperatures at the top/bottom).

---

## 3. Inference Phase: Loss Function

During inference (the Inverse Problem), the **model weights are frozen**. Instead, the **physical parameters $(\theta = \{Ra, Ha, Q, Da\})$ are treated as learnable variables** and optimized to match an observation.

### 3.1 Inference Loss Components
$$\mathcal{L}_{inference} = \mathcal{L}_{data} + \alpha \mathcal{L}_{physics} + \beta \mathcal{L}_{consistency} + \gamma \mathcal{L}_{boundary}$$

1.  **Observation Match ($\mathcal{L}_{data}$)**:
    - Measures how well the model (with the current parameter guesses) reproduces the observed flow field.
2.  **Consistency Loss ($\mathcal{L}_{consistency}$)**:
    - **Unique Logic**: This loss takes the predicted flow field and *algebraically solves* the governing equations for the parameters (e.g., deriving $Ra$ from the Momentum equation).
    - It then minimizes the difference between this "derived" parameter and the current "guess" parameter.
    - **Purpose**: Provides a direct gradient signal for the parameters, significantly speeding up convergence in the inverse problem.
3.  **Physics & Boundary Regularization**:
    - Ensures that the parameters being discovered result in a flow field that still obeys the laws of physics.

---

## 4. Design Rationale: Why Separate the Loss?

The separation between Training and Inference loss is fundamental to the **Surrogate-Inversion** workflow:

### 4.1 Training Strategy: Learning the Mapping
In the training phase, the parameters are **known (labels)**. The goal is to train a neural network that acts as a fast simulator (Surrogate). By including Physics Loss here, we force the network to learn not just "pixel patterns" but the underlying dynamics of the fluid.

### 4.2 Inference Strategy: Parameter Discovery
In the inference phase, the parameters are **unknown (targets)**. 
- We cannot use standard backpropagation to find parameters directly if the model hasn't been designed to be "invertible." 
- By freezing the model and optimizing the input parameters, we turn the inference into an optimization task. 
- The addition of **Consistency Loss** is critical here; while MSE loss only tells the model "the field looks wrong," Consistency Loss tells the model "the $Ra$ value is mathematically inconsistent with this velocity gradient," providing a much more precise correction path.

### 4.3 Computational Efficiency
- **Training** is done once on a large dataset to create a "physics-aware" model.
- **Inference** can then be performed on new, unseen observations very quickly (within seconds or minutes) without needing to re-train the entire network, making it suitable for real-time monitoring or system identification.
