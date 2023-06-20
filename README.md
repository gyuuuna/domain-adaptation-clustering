# Task-Adaptation Improvement with Data Augmentation
This repository provides an implementation for improving task-adaptation by augmenting data using domain-adapted sentence embeddings. The method described here focuses on embedding-based domain adaptation using DAPT (Domain-Adaptive Pretrained Transformer) on sentence embeddings. The main idea is to leverage contrastive loss and triplet loss to differentiate whether two sentences belong to the same label, rather than focusing on specific labels.

## Data downloads
+ task data (ag_news) : https://github.com/allenai/dont-stop-pretraining
  + `curl -Lo train.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/ag/train.jsonl`
  + `curl -Lo dev.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/ag/dev.jsonl`
  + `curl -Lo test.jsonl https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/ag/test.jsonl`
+ non task data (cnn_news) : https://www.kaggle.com/datasets/hadasu92/cnn-articles

## DAPT on Embedding
To perform domain adaptation on sentence embeddings, the following steps are followed:

1. Sentence Embedding: Train a Siamese model with GloVe embeddings using a contrastive loss function. The Siamese model takes pairs of sentences and outputs their embeddings. This embedding process captures information about the domain (label) of the sentences.

2. Data Augmentation: Use the trained encoder to generate embeddings for both task-related data (with labels) and unlabeled domain data.

3. Clustering: Cluster the unlabeled domain data based on similarity to the task data embeddings. Select data points from the unlabeled domain data that have similar embeddings to the task data.

4. Grouping: Group the task data by labels and sample N queries from each label. Similarly, sample M queries from the unlabeled domain data for each label.

5. Selection: Select the top-K domain data records that are close to the selected task data records based on their embeddings.

The augmented data, obtained through clustering and selection, can be used to improve task-adaptation performance.

## Downstream Task: AGNEWS Topic Classification
The downstream task used in this repository is AGNEWS topic classification. The available data for this task is as follows:

- Training data: Approximately 130,000 labeled sentences.
- Development data: Approximately 10,000 labeled sentences.
- Test data: Approximately 10,000 labeled sentences.

## Model
For DAPT embeddings, a Siamese model with contrastive loss and GloVe embeddings is used. This model determines whether pairs of data samples belong to the same task across different domains.

For TAPT (Task-Adaptive Pretraining), a RoBERTa baseline model is used. It is pretrained on a large corpus of unlabeled text data. The pretrained RoBERTa model is then fine-tuned using the RoberaForSequenceClassification implementation provided by Hugging Face.
