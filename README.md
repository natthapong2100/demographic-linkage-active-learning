> **Note:**  
> This code is not runnable since the actual implementation requires a large dataset (≈500k version) with ~1.3 million rows.

---

## Data Flow of this program

1. **Data Exploration**  
2. **Data Preprocessing**  
3. **Indexing (Blocking)**  
4. **Comparison**  
5. **Classification**  
    In this project, we employ three Machine Learning model for experimentation:  
    - (a) **Passive Learning (PL)** — implemented using Random Forest  
    - (b) **Active Learning (AL)**  
    - (c) **Bayesian Optimisation (BO)**  

---

## Research Structure

This research is divided into **two main folders**, according to the data quality:

- **Clean**
- **Corrupted**

Each folder contains:
- **Default file** (minHashLSH threshold = `0.7`)  
- **Experiment files** (discussed further in Experiments section)

---


## Experiments

### 1. Different Threshold Levels
- `Thresholds` in this program means the minHashLSH threshold in the Indexing (Blocking) step
- Thresholds tested:  
  - `0.65`  
  - `0.70` (default)  
  - `0.80`  
  - `0.90` → *best result across both models (PL, AL)*  

---

### 2. Random Training Data Selection
- Randomly select subsets of the training data.  
- Apply to **PL model**.

---

### 3. Random Pair Blocking
- In the **Indexing (Blocking)** step, we use **random blocking** instead of standard blocking.

---

### 4. Random Sampling in AL
- Use **Random Sampling** as query strategy, instead of default **Uncertainty Sampling**.  

---

### 5. Bayesian Optimisation
- Start with **threshold = 0.9**.  
- Implement **Bayesian Optimisation (BO)** to find optimal parameters.  
- Train **Random Forest** with using those BO optimal parameters.  
- Evaluate performance.
