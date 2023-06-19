import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

base_url = ''

test_df = pd.read_csv(base_url+'AG_test.csv', header=None)
train_df = pd.read_csv(base_url+'AG_train.csv')
domain_df = pd.read_csv(base_url+'cnn_full.csv')

input_texts = train_df['text']
labels = train_df['label']

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
encoded_inputs = tokenizer(input_texts.tolist(), padding=True, truncation=True, return_tensors='pt')
labels = torch.tensor(labels) - 1

train_inputs, val_inputs, train_labels, val_labels = train_test_split(encoded_inputs['input_ids'],
                                                                    labels,
                                                                    random_state=42,
                                                                    test_size=0.2)
train_dataset = TensorDataset(train_inputs, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    model.train()

    for batch in train_dataloader:
        batch_inputs, batch_labels = batch
        outputs = model(input_ids=batch_inputs, labels=batch_labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print('.', end='')

    model.eval()

    with torch.no_grad():
        val_outputs = model(input_ids=val_inputs, labels=val_labels)
        val_loss = val_outputs.loss
        val_accuracy = (val_outputs.logits.argmax(dim=1) == val_labels).float().mean()

    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Training loss: {total_loss / len(train_dataloader)}')
    print(f'Validation loss: {val_loss}')
    print(f'Validation accuracy: {val_accuracy}')

model.save_pretrained('roberta_classification_model')
