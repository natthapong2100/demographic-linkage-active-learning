# Enhancing Demographic Linkage with Active Learning in Machine Learning  
*MSc Dissertation Project*  

**Author:** Natthapong Lueangphumphitthaya \
**Supervisor:** Özgür Akgün \
**School of Computer Science, University of St Andrews** \
**July 8, 2025**

## Project Description

- **Demographic linkage** refers to the procedure of identifying the same individual that appears on **multiple historical documents**.
- Traditionally, this linkage involved manual searches for related family members, **birth records** and **death
records** data, etc. It’s **difficult** for human expert to annotate every record to find the identity linkage.
- **Active Learning** is utilised to overcome the high cost of manually labeling abundant historical documents and
validated its efficiency by employing **Passive Learning** as a comparative benchmark.  
- Additionally, **Bayesian Optimisation (BO)** was implemented for hyperparameter tuning, enhancing model performance under challenging data quality conditions.  


## Data Flow of the Program

1. **Data Exploration**  
2. **Data Preprocessing**  
3. **Indexing (Blocking)**  
4. **Comparison**  
5. **Classification**  
    In this project, we employ three Machine Learning model for experimentation:  
    - (a) **Passive Learning (PL)** — implemented using Random Forest  
    - (b) **Active Learning (AL)**  
    - (c) **Bayesian Optimisation (BO)**  


## Research Structure

This research is divided into **two main folders**, according to the data quality:

- **Clean**
- **Corrupted**

Each folder contains:
- **Default file** (threshold = `0.7`)  
- **Experiment files** (discussed further below)


## Experiments

### 1. Different Threshold Levels
- `Thresholds` in this program means the LSH threshold in the Indexing (Blocking) step
- Thresholds tested:  
  - `0.65`  
  - `0.70` (default)  
  - `0.80`  
  - `0.90` → *best result across both models (PL, AL)*  


### 2. Random Training Data Selection
- Randomly select subsets of the training data.  
- Apply to **PL model**.


### 3. Random Pair Blocking
- In the **Indexing (Blocking)** step, we use **random blocking** instead of standard blocking.


### 4. Random Sampling in AL
- Use **Random Sampling** as query strategy, instead of default **Uncertainty Sampling**.  


### 5. Bayesian Optimisation
- Start with **threshold = 0.9**.  
- Implement **Bayesian Optimisation (BO)** to find optimal parameters.  
- Train **Random Forest** with using those BO optimal parameters.  
- Evaluate performance.

## Result
- **Clean dataset**: Passive Learning, Active Learning, and Bayesian Optimisation got the same F1 score at 0.9991.
- **Corrupted dataset**: At F1 score, Bayesian Optimisation (0.9962) outperforms both Active Learning (0.9935) and Passive Learning (0.9945).


> **Note:**  
> This code is NOT runnable since the actual implementation requires a large dataset (≈500k version) with ~1.3 million rows.
