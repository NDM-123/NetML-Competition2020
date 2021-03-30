# NetML Challenge 2020 - Methods for malware detection version - 7063210
This competition is forked from https://github.com/ACANETS/NetML-Competition2020

## Overview
Recent progress in AI, Machine Learning and Deep Learning has demonstrated tremendous success in many application domains such as games and computer vision. Meanwhile, there are challenges of proliferating data flows and increasing malicious traffic on today’s Internet that call for advanced network traffic analysis tools. In this competition, we challenge the participants to leverage novel machine learning technologies to detect malicious flows and/or distinguish applications in a fine-grained fashion among network flows.

We are pleased to announce the NetML Challenge 2020. Given a set of flow features extracted from the original packet capture (PCAP) files, the task is to provide an accurate prediction. 

This NetML Challenge 2020 is the 1st of the Machine Learning Driven Network Traffic Analytics Challenge. In this year’s challenge, a collection of 1,199,139 flows in three different datasets are given including detailed flow features and labels. We also provide simple APIs and baseline machine learning models to demonstrate the usage of the datasets and evaluation metrics. There are 7 tracks for specific analytics objectives.

Two different tasks, namely malware detection and traffic classification, are available in this challenge. We provide two separate malware detection datasets, namely NetML and CICIDS2017, and one dataset for traffic classification: non-vpn2016.

- NetML dataset is constructed by selecting several PCAP files from www.stratosphereips.org website.
- CICIDS2017 dataset is generated using https://www.unb.ca/cic/datasets/ids-2017.html
- non-vpn2016 dataset is the subset of ISCX-VPN-nonVPN2016 dataset from https://www.unb.ca/cic/datasets/vpn.html

For detailed description about the datasets please check our paper https://arxiv.org/abs/2004.13006

Data for each dataset are available under the data folder in this repository. For NetML and CICIDS2017 datasets, two different annotations are available: top-level and fine-grained. Top-level annotations are for binary classification, i.e. benign or malware. For fine-grained, each dataset has different malware types and participants are required to label each flow according to their malware classes. For non-vpn2016, three level of annotations are available: top-level, mid-level and fine-grained. Annotations on the training sets are publicly available.

This challenge is organized by the Laboratory of Advanced Computer Architecture and Network Systems (ACANETS) at the University of Massachusetts Lowell, and sponsored by Intel Corporation. The results will be announced at NetML Challenge Workshop in IJCAI 2020.

----------------

## Dates
- April 8, 2021                            NetML challenge is launched!
- April 15, 2021                     Submission deadline at 16:00 Israel time (Each team will present their results and approach (5min )
----------------


## Challenge Guidelines

We have divided each dataset into 3 splits, test-challenge set, test-std set, and training set. Test-challenge set and test-std set are constructed by randomly extracting 10% of each class under fine-grained labels and the remaining 80% is left as training set. Annotations for training set is publicly available, but annotations for test-std and test-challenge tests will be private.

There are 7 tracks, and each track has two phases such as *dev* and *challenge*, making 14 different phases. First 7 phases are for development and last 7 phases are for the challenge. Development phase leaderboard will be public for participants to compare their team with other while challenge phase leaderboard will be announced after the challenge deadline. Below we describe each phase.

### 1- NetML_toplevel_dev
You will use NetML training dataset to train you own model using top-level annotations (benign or malware) and submit your predictions on the NetML test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 2- NetML_finegrained_dev
You will use NetML training dataset to train you own model using fine-grained annotations (benign, adload, ransomware, etc..) and submit your predictions on the NetML test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 3- CICIDS2017_toplevel_dev
You will use CICIDS2017 training dataset to train you own model using top-level annotations (benign or malware) and submit your predictions on the CICIDS2017 test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 4- CICIDS2017_finegrained_dev
You will use CICIDS2017 training dataset to train you own model using fine-grained annotations (benign, webAttack, portScan, etc..) and submit your predictions on the CICIDS2017 test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 5- non-vpn2016_toplevel_dev
You will use non-vpn2016 training dataset to train you own model using top-level annotations (chat, audio, video, etc..) and submit your predictions on the non-vpn2016 test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 6- non-vpn2016_midlevel_dev
You will use non-vpn2016 training dataset to train you own model using mid-level annotations (facebook, skype, hangouts, etc..) and submit your predictions on the non-vpn2016 test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 7- non-vpn2016_finegrained_dev
You will use non-vpn2016 training dataset to train you own model using fine-grained annotations (facebook_audio, facebook_chat, skype_audio, skype_chat, etc..) and submit your predictions on the non-vpn2016 test-std set with a JSON file containing your results in the correct format described on the evaluation page.

### 8- NetML_toplevel_challenge
You are expected to use top-level annotations (benign or malware) and submit your predictions on the NetML test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

### 9- NetML_finegrained_challenge
You are expected to use fine_grained annotations (benign, adload, ransomware, etc..) and submit your predictions on the NetML test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

### 10- CICIDS2017_toplevel_challenge
You are expected to use top-level annotations (benign or malware) and submit your predictions on the CICIDS2017 test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

### 11- CICIDS2017_finegrained_challenge
You are expected to use fine-grained annotations (benign, webAttack, portScan, etc..) and submit your predictions on the CICIDS2017 test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

### 12- non-vpn2016_toplevel_challenge
You are expected to use top-level annotations (chat, audio, video, etc..) and submit your predictions on the non-vpn2016 test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

### 13- non-vpn2016_midlevel_challenge
You are expected to use mid-level annotations (facebook, skype, hangouts, etc..) and submit your predictions on the non-vpn2016 test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

### 14- non-vpn2016_finegrained_challenge
You are expected to use fine-grained annotations (facebook_audio, facebook_chat, skype_audio, skype_chat, etc..) and submit your predictions on the non-vpn2016 test-challenge set with a JSON file containing your results in the correct format described on the evaluation page.

----------------



The download page contains links to all NetML, CICIDS2017 and non-vpn2016 train, test-std and test-challenge data and associated annotations (for training set only). Please specify any and all external data used for training in the "method description" when uploading results to the evaluation server.

Results must be submitted to the MODEL by the challenge deadline please see description in the main document.

----------------

## Dataset Description
Three different dataset are provided in this challenge: NetML, CICIDS2017, non-vpn2016. A proprietary flow feature extraction library is utilized to extract several META, TLS, DNS and HTTP flow features from raw traffic. These features are described in flow_features.csv file.
 
The 10% of each class is allocated to test-challenge set, another 10% for test-std set, and the remaining 80% of the extracted flow feature set is kept for the traininig set which are stored in the three files, 0_test-challenge_set.json.gz, 1_test-std_set.json.gz, 2_training_set.json.gz. Every flow for each dataset has been assigned a unique flow id number. Every flow for NetML and CICIDS2017 datasets has two different annotations, top-level and fine-grained. Every flow in non-vpn2016 dataset has three different annotations, top-level, mid-level, fine-grained. Annotations, i.e. ground-truth are separately stored for each level of annotations.


### NetML
Raw traffic (e.g., Pcap files) is obtained from www.stratosphereips.org. This dataset has 20 types of attacks and normal traffic flows. 

The total number of flows for different splits:
- test-challenge set: 48,394 
- test-std set : 48,394
- traininig set: 387,268

### CICIDS2017
Raw traffic is obtained from https://www.unb.ca/cic/datasets/ids-2017.html. Attack flows are extracted by filtering each workday PCAP files with respect to time interval and IPs described in their webpage. The extracted dataset has 7 types of malware attacks and normal traffic flows.

The total number of flows for different splits:
- test-challenge set: 55,128 
- test-std set : 55,128
- traininig set: 441,116

### non-vpn2016
PCAP files are downloaded from https://www.unb.ca/cic/datasets/vpn.html. The original dataset has both vpn and non-vpn packet capture files but we only focus on non-vpn captures. In top-level annotation, we categorize the traffic into 7 groups: audio, chat, email, file_transfer, tor, video, P2P. In mid-level annotation, we group into 18 classes according to the application type such as aim_chat, facebook, hangouts, skype, youtube etc. In fine-level annotation, we treat each action as a different category and obtain 31 classes such as facebook_chat, facebook_video, skype_chat, skype_video etc.

The total number of flows for different splits:
- test-challenge set: 16,323 
- test-std set : 16,323
- traininig set: 131,065

----------------

## Quickstart Guideline

Train a simple Random Forest model using sklearn and create submission.json file.

### Requirements
- python 3
- numpy
- pandas
- sklearn
- matplotlib

We recommend you to use virtual environment. For details please see https://docs.python.org/3/tutorial/venv.html

```shell
$ git clone https://github.com/ACANETS/NetML-Competition2020
```

### Usage Example (alternative way is to use the official ipython)
Provided baseline script evaluates the model according to the validation set scores created using 20% of the training set. The remaining 80% of the training set is used to train the model.

To train for Phase-1 NetML_toplevel_dev and create a JSON file ready to submit:

```shell
$ python3 RF_baseline.py --dataset ./data/NetML --anno top --submit test-std
```

The above command runs the python script and creates a './results' folder in the main directory. You can find your models confusion matrix with F1 and mAP scores and submission_test.json file under the folder with the time you ran the above command.

This script uses the functions defined under ./utils/helper.py to read the data, display and save the confusion matrix, and create the submission json file.

- Load data and split the validation set:

```python
# Get training data in np.array format
Xtrain, ytrain, class_label_pair, Xtrain_ids = get_training_data(training_set, training_anno_file)

# Split validation set from training data
X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain,
                                                test_size=0.2, 
                                                random_state=42,
                                                stratify=ytrain)
```

The above function uses ./utils/featureDict_META.json file to parse the META features from json.gz files but you are encouraged to write your own parser to include other features as well!

- Preprocess for a better performance:

```python
# Preprocess the data
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

- Define and train the model. Print training and validation set accuracy:

```python
# Train RF Model
print("Training the model ...")
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs = -1, max_features="auto")
clf.fit(X_train_scaled, y_train)

# Output accuracy of classifier
print("Training Score: ", clf.score(X_train_scaled, y_train))
print("Validation Score: ", clf.score(X_val_scaled, y_val))
```

- Plot the confusion matrix and save under ./results/<%Y%m%d-%H%M%S>/CM.png:

```python
# Print Confusion Matrix
ypred = clf.predict(X_val_scaled)

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plot_confusion_matrix(directory=save_dir, y_true=y_val, y_pred=ypred, 
                        classes=class_names, 
                        normalize=False)
```

- If you want to create a submission json file for a selected set, the following script handles it and saves the selected json under ./results/<%Y%m%d-%H%M%S>/ folder:

```python
# Make submission with JSON format
if args.submit == "test-std" or args.submit == "both":
    submit(clf, test_set, scaler, class_label_pair, save_dir+"/submission_test-std.json")
if args.submit == "test-challenge" or args.submit == "both"
    submit(clf, challenge_set, scaler, class_label_pair, save_dir+"/submission_test-challenge.json")
```

----------------

### Evaluation
Submission file should be as the following JSON format:

```python
{"id1": "label1", "id2": "label2", ...}
```

For multi-class classification problems, we use F1 score and mean average precision (mAP) as two different metrics.

F1=2\*precision\*recall/(precision+recall)

where:

- precision=TP/(TP+FP)
- recall=TP/(TP+FN)

- True Positive [TP] = your prediction is 1, and the ground truth is also 1 - you predicted a positive and that's true!
- False Positive [FP] = your prediction is 1, and the ground truth is 0 - you predicted a positive, and that's false.
- False Negative [FN] = your prediction is 0, and the ground truth is 1 - you predicted a negative, and that's false.

mAP = 1/N\*(\sum(AP<sub>i</sub>))

where:

- N: number of classes
- AP: average precision for each class

'overall' score is bounded between 0 and 1 and higher value represents more accurate results. It is calculated by multiplying the two scores: 

overall = F1\*mAP

----------------

For binary classifications, in other words detection problems, we use True Positive Rate (TPR) as detection rate of malware and False Alarm Rate as two metrics.

- TPR = TP/(TP+FN)
- FAR = FP/(TN+FP)

'overall' score is bounded between 0 and 1 and higher value represents more accurate results. It is calculated by multiplying detection rate with the 1-FAR: 

overall = TPR\*(1-FAR)

----------------

## Directory Structure
* NetML-Competition2020
    - readme.md
    - RF_baseline.py
    - NetML-competition.ipynb
    - flow_features.csv
    * data
        - labels.txt
        * CICIDS2017
            - ...
        * NetML
            - ...
        * non-vpn2016
            - ...
    * results
        * CICIDS2017_fine
            - CM.png
        * CICIDS2017_top
            - CM.png
        * NetML_fine
            - CM.png
        * NetML_top
            - CM.png
        * non-vpn2016_fine
            - CM.png
        * non-vpn2016_mid
            - CM.png
        * non-vpn2016_top
            - CM.png
    * utils
        - featureDict_META.json
        - helper.py

----------------

## License

BSD
