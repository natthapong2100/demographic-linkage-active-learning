Note: this code is not runnable since all of the actual implementation require the big dataset (500k ver.) with approximately 1.3 million rows.

This research is divided into 2 folder; clean and corrupted folder. Then, the following file is separated into the default file (0.7 threshold) and experiment file.

1. Different level threshold
- Listing from 0.65, 0.7 (default), 0.8, 0.9 (the best in all models; PL, AL)

2. Randomly selected the training data
- Randomly selected the training data into PL model.

3. Random Pair blocking
- In the Indexing (Blocking) step, do the random instead.

4. Random sampling in AL
- Use Random Sampling as Query strategy in modAL, instead of Uncertainty sampling.

5. Bayesian Optimisation
- Grab the 0.9 threshold level and implement the BO model. After got the optimal params from BO model, then train into the Random Forest to achive the result.
