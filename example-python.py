
import pandas as pd


df = pd.read_csv(r"../Data/3k/_1/corrupted/_1/birth_records.csv")


print(df.shape)


print(df.columns)


df.count()


df = df[:3000]


df.describe()


print(df.info())


# To check the total number of null values in each column
print(df.isnull().sum())


#df['unique_id'] = range(1, len(df) + 1)able with your dataset (ensure it's a 2D array or DataFrame with numerical values). The principal_components will hold the result of PCA with the 
original_df = df.copy()


original_df.iloc[0]['family']


#columns_to_keep = ['ID']
columns_to_keep = ["sex","father's forename","father's surname",
"father's occupation",
"mother's forename",
"mother's maiden surname",
"mother's occupation",
"day of parents' marriage",
"month of parents' marriage",
"year of parents' marriage",
"place of parent's marriage"]
#df = df.filter(rows_to_keep)
df = df.loc[:, columns_to_keep].copy()


#df.drop(['ID'], axis=1, inplace=True)


df


original_df


print(original_df.iloc[2067]['family'])


cols = df.columns


for colname in cols:
    print(str(colname))


# ## Indexing
# ### Blocking
# - MinHash


from datasketch import MinHash
from datasketch import MinHashLSH

minhash_dict = {}

# Iterate over the dataset and generate MinHash signatures
for idx, row in df.iterrows():
    minhash = MinHash()
    # relevant columns for similarity comparison
    for colnames in cols:
        data_to_hash = str(row[colnames]) #using all the features
    for token in data_to_hash.split():
        minhash.update(token.encode('utf8'))
    minhash_dict[idx] = minhash



# Initialize the MinHash LSH index
lsh = MinHashLSH(threshold=0.9)

# Insert MinHash objects into the index
for idx, minhash in minhash_dict.items():
    lsh.insert(idx, minhash)

blocks = []

# Iterate over each record in the dataset
for idx, row in df.iterrows():
    minhash = MinHash()
    # Include the relevant columns for blocking
    for colnames in cols:
        data_to_query = str(row[colnames]) #using all the features

    for token in data_to_query.split():
        minhash.update(token.encode('utf8'))
    
    # Query the MinHash LSH index to retrieve similar records
    similar_records = lsh.query(minhash)
    
    # Create a block and add the similar records
    block = {'record_id': idx, 'similar_records': similar_records}
    blocks.append(block)



blocksK = blocks[0:15]
for block in blocksK:
    print("Block:")
    print("Record ID:", block['record_id'])
    print("Similar Records:", block['similar_records'])
    print("---")



print(len(blocks))


num = 0
print(len(blocks[num]['similar_records']))
# simillar records at block 0


# ## Comparing metrics
# - jaccard, jaro winkler, damerau_levenshtein_distance


def jaccard_similarity(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    intersection_length = len(set1 & set2)
    union_length = len(set1 | set2)
    return intersection_length / union_length

import pandas as pd
import splink
from jellyfish import jaro_distance, jaro_winkler, levenshtein_distance, damerau_levenshtein_distance


#---------

outer_array=[]
distance_dict={}

#should contain record_id, distance_data for each of the blocks being stored in another 
#distances_data = pd.DataFrame(columns=['record_id','jaro_distance', 'jaro_winkler_distance', 'levenshtein_distance', 'damerau_levenshtein_distance', 'jaccard_distance'])
distances_data = pd.DataFrame(columns=['record_id','jaro_winkler_dist', 'damerau_levenshtein_dist', 'jaccard_dist'])
for block in blocks:
    inner_array=[]
    #print( "block")
    #print(block['record_id'])
    for similar in block['similar_records']:
        #if(block['record_id'] == similar):
         #   continue
        #else:
        each_block_dict={}
        each_feature = {}
        #distance_dict.update({block['record_id']:distance_array})
        for features in cols:
            
            record1 = str(df.loc[block['record_id']][features])
            record2 = str(df.loc[similar][features])

            #finding the distances
            #jaro_dist = jaro_distance(record1, record2)


            # Calculate Jaro-Winkler distance
            jaro_winkler_dist = jaro_winkler(record1, record2)
            
            # Calculate Levenshtein distance
            #levenshtein_dist = levenshtein_distance(record1, record2)
        
            #damerau_levenshtein_distance
            damerau_levenshtein_dist = damerau_levenshtein_distance(record1, record2)
            
            #jaccard_similarity
            jaccard_dist = jaccard_similarity(record1, record2)
            
            #storing the distances for each feature
            each_feature.update({features : [
                #jaro_dist, 
                jaro_winkler_dist, 
                damerau_levenshtein_dist,
                jaccard_dist
                #, 
                #damerau_levenshtein_dist
            ]})
            
        each_block_dict.update({similar:each_feature}) # measure similarity by metrics (jaro winkler, damerau_levenshtein, and jaccard distance) in each block 
            
        inner_array.append(each_block_dict)
    outer_array.append(inner_array)


print(len(inner_array))


print(outer_array[0])


print(len(outer_array))


#creating a labelling array
data_array=[]
#label_array=[]

for block in blocks:
    label_array=[]
    #count = 0
    for similar in block['similar_records']:
        if((original_df.loc[block['record_id']]['family'] == original_df.loc[similar]['family'])):
        #if((original_df.loc[idx]['ID'] == original_df.loc[similar]['ID'])):
            label_array.append(1)
        else:
            label_array.append(0)
    data_array.append(label_array)


data_array[0]


result = []  # Store the counts for each subarray
for subarray in data_array:
    count = 0
    for element in subarray:
        if element == 1:
            count += 1
    result.append(count)



result


print(len(data_array))


#counting = 0
#for elements in data_array:
#    for each in elements:
#        if(each == 1):
#            counting+=1
#print(counting)


zer = 0
for element in result:
    if element == 0:
        zer += 1
print(zer)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

# Flatten outer_array and convert it to a feature matrix (X)

X = []
parent_idx = 0
for inner_array in outer_array:
    for each_block_dict in inner_array:
        
        for blocked_idx, distances in each_block_dict.items():
            eachrow=[]
            eachrow.append(parent_idx)
            eachrow.append(blocked_idx)
            for feature_distances in distances.values():
                for each_val in feature_distances:
                    #keeping the first column as a the parent index
                    #keeping the second column as the child blocked index
                    #eachrow
                    eachrow.append(each_val)
            X.append(eachrow)
    parent_idx += 1
X = np.array(X)




X


# Convert data_array to the target labels (y)
y = []
for label_array in data_array:
    y.extend(label_array)
y = np.array(y)


#checking the data weightage -- leakage
unique_values, counts = np.unique(y, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"{value}: {count}")


from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling,entropy_sampling,margin_sampling
from modAL.batch import uncertainty_batch_sampling
from modAL.disagreement import consensus_entropy_sampling, max_disagreement_sampling,max_std_sampling, vote_entropy_sampling
from modAL.multilabel import SVM_binary_minimum, max_loss
from imblearn.over_sampling import SMOTE


# 1. Randomly initialize training set
#initial_training_indices = np.random.choice(len(X), size=50, replace=False) #i think i should change this to a percentage
#reomving the first two columns as it is the index numbreing to the original
#X_initial_with_index = X[initial_training_indices]
#y_initial = y[initial_training_indices]

#X_initial = X_initial_with_index[:,2:]

#print(len(initial_training_indices))




import numpy as np
from sklearn.datasets import make_classification

feature_size = X[:,2:].shape[1] #need to remove the first two columns

X_uncertainty = X #creating a copy of the data
X_uncertainty_sampling = X_uncertainty[:, 2:]

from sklearn.preprocessing import StandardScaler

# Assuming X_Data is your combined array of distance metrics
scaler = StandardScaler()
X_uncertainty_sampling = scaler.fit_transform(X_uncertainty_sampling)


X_uncertainty_sampling




# Generate synthetic imbalanced dataset
#X, y = make_classification(n_samples=1000, n_features=20, weights=[0.95], random_state=42)

#positive_indices = np.where(y == 1)[0]
#negative_indices = np.where(y == 0)[0]

# Randomly select 500 indices for each class
#positive_sample_indices = np.random.choice(positive_indices, size=500, replace=False)
#negative_sample_indices = np.random.choice(negative_indices, size=500, replace=False)

# Combine the selected indices for both classes
#initial_training_indices = np.concatenate((positive_sample_indices, negative_sample_indices))


# Use initial_training_indices for your initial training set
initial_training_indices = np.random.choice(len(X), size=5, replace=False)
X_initial = X_uncertainty_sampling[initial_training_indices]
y_initial = y[initial_training_indices]

# Delete selected data from the dataset
X_clean_us = np.delete(X_uncertainty_sampling, initial_training_indices, axis=0)
y_us = np.delete(y, initial_training_indices)

# Calculate class distribution using np.unique
unique_values, counts = np.unique(y_initial, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"{value}: {count}")

if len(unique_values) > 1:
    majority_class = unique_values[np.argmax(counts)]
    minority_class = unique_values[1 - np.argmax(counts)]
    class_ratio_initial = counts[minority_class] / counts[majority_class]
else:
    majority_class = unique_values[np.argmax(counts)]
    minority_class = 0
# Calculate class imbalance ratio


class_ratio_initial = 0
print("Initial Training Set Class Imbalance Ratio:", class_ratio_initial)



from imblearn.over_sampling import SMOTE

# Calculate desired minority-to-majority ratio (7:10)
desired_ratio = 1 / 10

# Calculate the desired number of minority samples based on the ratio
#desired_minority_samples = int(desired_ratio * counts[majority_class])

#n_neighbors = min(3, 6) 

# Apply SMOTE to the original data
#smote = SMOTE(sampling_strategy={minority_class: desired_minority_samples}, random_state=42)
# Calculate oversampling factor for the minority class
#oversampling_factor = int(desired_minority_samples / counts[minority_class])

# Oversample the minority class to achieve the desired ratio
#smote = SMOTE(sampling_strategy=0.5,k_neighbors=2, random_state=42)
#X_synthetic, y_synthetic = smote.fit_resample(X_initial, y_initial)
#smote = SMOTE(sampling_strategy={minority_class: desired_minority_samples}, k_neighbors=3, random_state=42)
#X_synthetic, y_synthetic = smote.fit_resample(X_initial, y_initial)

# Calculate class distribution in the synthetic dataset
#unique_values_synthetic, counts_synthetic = np.unique(y_synthetic, return_counts=True)
#for value, count in zip(unique_values_synthetic, counts_synthetic):
#    print(f"{value}: {count}")
    
# Calculate class imbalance ratio in the synthetic dataset
#class_ratio_synthetic = counts_synthetic[minority_class] / counts_synthetic[majority_class]
#print("Synthetic Data Set Class Imbalance Ratio:", class_ratio_synthetic)

X_synthetic = X_initial
y_synthetic = y_initial


# ## RandomForest




# Create the base classifier
base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.98, random_state=42)




#clean X and take the data from the the whole X

num_features = X_clean_us.shape[1]
X_train_us = np.empty((0, num_features))  # Initialize an empty array with the expected number of features to store the training data
y_train_us = []


print(y_us.shape[0])
print()
print(X_clean_us.shape[0])


import os
import csv
import random
from sklearn.metrics import confusion_matrix

# Define the base directory where you want to create the experiment folder
base_dir = "experiment_data"

i = 1
# Create a random name for the experiment folder
experiment_folder = os.path.join(base_dir, "experiment_" + str(i))

# Check if the experiment folder already exists
while os.path.exists(experiment_folder):
    i += 1
    experiment_name = "experiment_" + str(i)
    experiment_folder = os.path.join(base_dir, experiment_name)
print('experiment ', i)
# Create the experiment folder if it doesn't exist
if not os.path.exists(experiment_folder):
    os.makedirs(experiment_folder)

# Define the path for your log file within the experiment folder
log_filename = os.path.join(experiment_folder, "experiment_log_uncertainty.csv")




# ## Active Learner and Metric Performance


# Open the log file for writing the experiment data
with open(log_filename, "w", newline="") as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Queries", "Precision", "Recall","F1", "max_probability"])
    #defining the simulation counter.
    sim_counter = 1 #------------------------------------------------------------------------------
    user_input = 3 #---------------------------------------------------------------------------------
    halt = None

    #the uncertainty checking part
    current_max_probability = float('inf')  # Initialize with a large value
    previous_max_probability = float('-inf')  # Initialize with a small value
    
    # 2. Initialize ActiveLearner
    learner = ActiveLearner(
        estimator=base_classifier,
        query_strategy=uncertainty_sampling,
        X_training=X_synthetic,
        y_training=y_synthetic
    )
    
    queries_set_checked = 0
    # 3. Active Learning Loop
    target_f1 = 1.0
    current_f1 = 0.0
    test_s = 0.9

    #----------------------------------------------
    while True: #or current_f1 > target_f1:

        if user_input == 1:
            halt = int(input("Continue labeling each on your own: (Yes - 1), (No - 0))"))
        if halt == 0:
            user_input = 2
        
        if user_input is None or sim_counter == 0:
            if X_train_us.shape[0] > 0: #just checking if there is any data in the set
                # Train the model with the updated training set
                learner.teach(X_train_us, y_train_us)                
                
                #predict and update
        
                # Calculate F1 score using a validation dataset
                y_pred = learner.predict(X_clean_us)
                current_f1 = f1_score(y_us, y_pred)
                # Calculate precision and recall
                precision = precision_score(y_us, y_pred)
                recall = recall_score(y_us, y_pred)
                queries_set_checked += 1
                #storing the iteration and the data
                # Write the experiment data to the log file
                log_writer.writerow([queries_set_checked, precision, recall, current_f1, max_probability])
                log_file.flush()  # Flush the buffer to ensure data is written immediately

                print("------------------------------------------------------------------------------")
                print(f"Current F1 score: {current_f1:.4f}")
                print(f"Current precision: {precision:.4f}")
                print(f"Current Recall: {recall:.4f}")
                c_matrix = confusion_matrix(y_us, y_pred)
                conf_matrix_0 = confusion_matrix(y_us, y_pred, labels=[0])
                conf_matrix_1 = confusion_matrix(y_us, y_pred, labels=[1])
                print("confusion matrix")
                print(c_matrix)
                print("Confusion for class 0:")
                print(conf_matrix_0)
                print("Confusion for class 1:")
                print(conf_matrix_1)
                print("==============================================================================")
                X_train_us = np.empty((0, num_features))
                y_train_us = np.empty(0)
                if current_f1 >= target_f1:
                    break
                    
            #user_input = int(input('''Enter an option:
            #Note:This dialogue pops up after every 100 labelings if in simulation.
            #Enter:
            #1) If you want to do the next 100 labelings.(You can stop and enter the simulation anytime)
            #2) If you want the simulation to do the next 100 labelings
            #3) If you want the simulations to do all the labelings
            #'''))
            user_input = 3
            sim_counter = 1
    
    
        if(user_input == 3):
            user_input = 0


        query_idx, query_instance = learner.query(X_clean_us)

        # Calculate maximum predicted probability for the selected data point
        probs = learner.predict_proba(query_instance.reshape(1, -1))
        max_probability = np.max(probs)
        #current_max_probability = max_probability
        print('max probability', max_probability) #need to break if its lower
        print('current_query', queries_set_checked)
        #if max_probability < previous_max_probability:
         #   current_max_probability = max_probability
          #  break
        #else:
        #    previous_max_probability = current_max_probability
         #   current_max_probabilty = max_probability
            

        print('current max probability', max_probability) #need to break if its lower
        print('query num', sim_counter)
        #The code to find the datapoints that are to be compared

        #specific_data = blocks[block_idx][similar_idx]
        data1 = int((X_uncertainty_sampling[query_idx,0])[0])
        data2 = int((X_uncertainty_sampling[query_idx,1])[0]) #which i dont completely get it        

        if user_input == 1:
            # Display the queried instance to the user
            print(df.iloc[data1])  
            print('---------------------------------------------------------------------------------------------')
            #Display the second data
            print(df.iloc[data2])
            
            # Ask the user to label the queried instance
            query_label = int(input(f"Label the instance at index (0 or 1 -> 1 if similar): "))
            sim_counter -= 1
        elif user_input == 2 or user_input == 0:
            sim_counter -= 1
            query_label = y_us[query_idx]
            #print(original_df.iloc[data1]['family'])
            #print(original_df.iloc[data2]['family'])
            #if((original_df.iloc[data1]['family']) == (original_df.iloc[data2]['family'])):
                #query_label = 1
            #else:
                #query_label = 0
        
        # Update the training set with the queried instance and its label
        X_train_us = np.vstack((X_train_us, query_instance))
        y_train_us = np.append(y_train_us, query_label)

        #deleting the data from X
        X_clean_us = np.delete(X_clean_us, query_idx, axis=0)
        y_us = np.delete(y_us, query_idx)
        #print('size')
        #print(y_us.shape[0])
        #print()
        #print(X_clean_us.shape[0])
        
    #------------------------------------
    y_pred = learner.predict(X_clean_us)
    precision = precision_score(y_us, y_pred)
    recall = recall_score(y_us, y_pred)
    f1 = f1_score(y_us, y_pred)
    print("Precision micro:", precision)
    print("Recall micro:", recall)
    print("F1 Score micro:", f1)
    print()



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

#plotting the AUC-ROC curve

# Calculate the false positive rate (fpr), true positive rate (tpr), and AUC
fpr, tpr, threshold = roc_curve(y_us, y_pred)
roc_auc = auc(fpr, tpr) #Area under the curve

# Plot the ROC curve
plt.figure(figsize=(8, 6), dpi =100)
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
#plt.xlim([0.0, 1.05])  # Increase the upper y-limit to provide some space for the plot
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Get the experiment folder and log filename
experiment_folder = os.path.join(base_dir, experiment_name)

# Save the plot in the experiment folder
plot_filename = os.path.join(experiment_folder, "roc_curve.png")
plt.savefig(plot_filename)


plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_us, y_pred)

# Calculate PR AUC
pr_auc = auc(recall, precision)

# Plot the PR curve
plt.figure(figsize=(8, 6), dpi=100)
plt.plot(recall, precision, color='darkorange', label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

# Get the experiment folder and log filename
experiment_folder = os.path.join(base_dir, experiment_name)

# Save the plot in the experiment folder
plot_filename = os.path.join(experiment_folder, "pr_curve.png")
plt.savefig(plot_filename)


plt.show()






