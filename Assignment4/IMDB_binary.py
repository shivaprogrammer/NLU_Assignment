#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMDB (binary class) dataset:


# In[2]:


import torch
print(torch.__version__)
print(torch.cuda.is_available())


# In[3]:


import pandas as pd
import spacy
import numpy as np


# In[4]:


from collections import Counter
import re
nlp = spacy.load("en_core_web_sm")


# In[5]:


import os
import pandas as pd

def load_imdb_data(directory):
    data = []
    for label_type in ['pos', 'neg']:
        dir_path = os.path.join(directory, label_type)
        label = 1 if label_type == 'pos' else 0
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    data.append((text, label))
    return pd.DataFrame(data, columns=['text', 'label'])

# Paths
train_path = "IMDB (binary class) dataset/train"
test_path = "IMDB (binary class) dataset/test"

# Load raw train and test data
full_train_df = load_imdb_data(train_path)
test_df = load_imdb_data(test_path)

full_train_df = full_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split last 10% for dev
split_index = int(len(full_train_df) * 0.9)
train_df = full_train_df[:split_index].reset_index(drop=True)
dev_df   = full_train_df[split_index:].reset_index(drop=True)

print(f"Train set size: {len(train_df)}")
print(f"Dev set size:   {len(dev_df)}")
print(f"Test set size:  {len(test_df)}")


# In[6]:


train_df


# In[7]:


dev_df.shape


# In[8]:


for col in test_df.columns:
    print(col)


# In[9]:


print(train_df['label'].value_counts())
#label_map = {0: "negative", 1: "positive"}


# In[10]:


# Basic cleaning
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", '', text)
    text = re.sub(r"<br\s*/?>", " ", text)  

    return text.strip()

# Tokenize
def tokenize(text):
    return [token.text.lower() for token in nlp(text) if not token.is_space]

# Encode with word2idx
def encode_tokens(tokens, word2idx):
    return [word2idx.get(token, word2idx['UNK']) for token in tokens]

# Pad to max_len
def pad_or_truncate(seq, max_len, pad_idx):
    if len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq + [pad_idx] * (max_len - len(seq))


# In[11]:


# One-hot encode
def one_hot_encode_sequence(encoded_seq, vocab_size):
    one_hot = np.zeros((len(encoded_seq), vocab_size), dtype=np.float32)
    for i, idx in enumerate(encoded_seq):
        one_hot[i, idx] = 1.0
    return one_hot

# --- STEP 1: Process TRAIN set ---
train_df['text'] = train_df['text'].astype(str).apply(clean_text)
train_df['tokens'] = train_df['text'].apply(tokenize)

# Build vocab
all_tokens = [token for tokens in train_df['tokens'] for token in tokens]
token_freq = Counter(all_tokens)
vocab = ['PAD', 'UNK'] + sorted([word for word, freq in token_freq.items() if freq >= 5])
word2idx = {word: idx for idx, word in enumerate(vocab)}

print("Vocabulary size (incl. PAD & UNK):", len(vocab))
print("Sample vocab:", vocab[:10])

# Encode and pad train
train_df['encoded_tokens'] = train_df['tokens'].apply(lambda t: encode_tokens(t, word2idx))
max_len = train_df['encoded_tokens'].apply(len).sum() // len(train_df)
pad_idx = word2idx['PAD']
train_df['padded'] = train_df['encoded_tokens'].apply(lambda x: pad_or_truncate(x, max_len, pad_idx))
train_df['onehot'] = train_df['padded'].apply(lambda x: one_hot_encode_sequence(x, len(vocab)))

# --- STEP 2: Reusable function for DEV/TEST ---
def preprocess_df(df, word2idx, max_len, pad_idx, vocab_size):
    df['text'] = df['text'].astype(str).apply(clean_text)
    df['tokens'] = df['text'].apply(tokenize)
    df['encoded_tokens'] = df['tokens'].apply(lambda t: encode_tokens(t, word2idx))
    df['padded'] = df['encoded_tokens'].apply(lambda x: pad_or_truncate(x, max_len, pad_idx))
    df['onehot'] = df['padded'].apply(lambda x: one_hot_encode_sequence(x, vocab_size))
    return df

dev_df = preprocess_df(dev_df, word2idx, max_len, pad_idx, len(vocab))
test_df = preprocess_df(test_df, word2idx, max_len, pad_idx, len(vocab))

print("Max length used:", max_len)
print("Train shape:", train_df['onehot'].iloc[0].shape)
print("Dev shape:", dev_df['onehot'].iloc[0].shape)


# In[12]:


# Show vocab stats
print("\n--- Vocabulary Stats ---")
print("Total unique tokens in training data:", len(token_freq))
print("Words kept (freq >= 5):", len(vocab) - 2)
print("Words dropped (replaced with UNK):", len(token_freq) - (len(vocab) - 2))


# In[13]:


print("\n--- Sample Rows from Train Data ---")
for i in range(3):
    print(f"\nOriginal Text: {train_df['text'].iloc[i]}")
    print(f"Tokens: {train_df['tokens'].iloc[i]}")
    print(f"Encoded Tokens: {train_df['encoded_tokens'].iloc[i]}")
    print(f"Padded: {train_df['padded'].iloc[i]}")
    print(f"One-hot Shape: {train_df['onehot'].iloc[i].shape}")


# In[14]:


label_map = {0: "negative",  1: "positive"}
train_df["label_name"] = train_df["label"].map(label_map)

print("\n--- Label Distribution in Train ---")
print(train_df["label_name"].value_counts())

print("\n--- Label Distribution in Dev ---")
dev_df["label_name"] = dev_df["label"].map(label_map)
print(dev_df["label_name"].value_counts())

print("\n--- Label Distribution in Test ---")
test_df["label_name"] = test_df["label"].map(label_map)
print(test_df["label_name"].value_counts())


# In[15]:


import numpy as np
np.set_printoptions(threshold=100, linewidth=160)  

# View one example from train_df
sample_i = 0  
sample_onehot = train_df['onehot'].iloc[sample_i]
sample_tokens = train_df['tokens'].iloc[sample_i]
sample_encoded = train_df['encoded_tokens'].iloc[sample_i]
sample_padded = train_df['padded'].iloc[sample_i]

print(f"\n Original Text:\n{train_df['text'].iloc[sample_i]}")
print(f"\n Tokens:\n{sample_tokens}")
print(f"\n Encoded Tokens:\n{sample_encoded}")
print(f"\n Padded Sequence:\n{sample_padded}")
print(f"\n One-Hot Shape: {sample_onehot.shape}")
print(f"\n One-Hot Vector for First 3 Tokens:")

# Show the one-hot vector of first 3 tokens
for idx, (token, vec) in enumerate(zip(sample_tokens[:3], sample_onehot[:3])):
    print(f"{token} → one-hot[{idx}]: {vec.nonzero()[0][0]} → {vec}")


# In[16]:


unk_count = sum([1 for token in train_df['encoded_tokens'].explode() if token == word2idx['UNK']])
print("Total UNK tokens in training data:", unk_count)


# In[17]:


'''
 Architecture Design
We're building a 3-layer Feed-Forward Neural Network (FFNN) for multi-class sentiment classification (positive, negative, neutral). The input is a padded + one-hot encoded sentence tensor.

 Structure:
Input Layer
Shape: [batch_size, max_len, vocab_size]
→ Flattened to: [batch_size, max_len × vocab_size]

Hidden Layer 1

Size: 256 units

Activation: ReLU

Hidden Layer 2

Size: 128 units

Activation: ReLU

Output Layer

Size: 3 units (for positive, neutral, negative)

Activation: None (logits) → used with CrossEntropyLoss
'''


# In[18]:


import torch.nn as nn


# In[19]:


class FeedForwardSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden1=256, hidden2=128, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2idx['PAD'])
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(embedding_dim * max_len, hidden1)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = F.relu(self.dropout1(self.fc1(x)))
        x = F.relu(self.dropout2(self.fc2(x)))
        return self.output(x).squeeze(1)


# In[20]:


from torch.utils.data import Dataset, DataLoader

class IMDBDataset(Dataset):
    def __init__(self, df):
        self.inputs = df["padded"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)  
        return x, y


# In[21]:


train_dataset = IMDBDataset(train_df)
dev_dataset   = IMDBDataset(dev_df)
test_dataset  = IMDBDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader   = DataLoader(dev_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)


# In[22]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[23]:


import torch.nn.functional as F


# In[24]:


print("Train label range:", train_df["label"].unique())
print("Dev label range:", dev_df["label"].unique())
print("Test label range:", test_df["label"].unique())


# In[25]:


import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
EPOCHS = 10
best_dev_acc = 0.0
train_losses = []
dev_accuracies = []
import torch.optim as optim


from tqdm.notebook import tqdm 
model = FeedForwardSentiment(vocab_size=len(word2idx))  
model = model.to(device) 
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()


for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

    for inputs, labels in loop:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)


    # === Validation ===
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dev_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    dev_acc = accuracy_score(all_labels, all_preds)
    dev_accuracies.append(dev_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {avg_train_loss:.4f} — Dev Accuracy: {dev_acc:.4f}")

    # Save best model
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        torch.save(model.state_dict(), "Binary_best_ffnn_model.pt")
        print(" Best model saved!")


# In[26]:


pad_idx = word2idx['PAD']
num_pad_tokens = sum([token == pad_idx for padded in train_df['padded'] for token in padded])
total_tokens = sum([len(padded) for padded in train_df['padded']])
print("PAD ratio:", num_pad_tokens / total_tokens)


# In[27]:


print("Sample outputs (logits):", outputs[:5].detach().cpu().numpy())
print("Sample preds (after sigmoid):", torch.sigmoid(outputs[:5]).detach().cpu().numpy())


# In[28]:


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o')
plt.title("Train Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(dev_accuracies, marker='o', color='green')
plt.title("Dev Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()


# In[29]:


# Load best model
model.load_state_dict(torch.load("Binary_best_ffnn_model.pt"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)  # raw logits
        preds = (torch.sigmoid(outputs) > 0.5).long()  # binary prediction (0 or 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Overall metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

print("\n Final Test Set Evaluation:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Per-class metrics
precision_all, recall_all, f1_all, support = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1])

print("\n Per-class Metrics:")
for i, label in enumerate([0, 1]):
    label_str = "negative" if label == 0 else "positive"
    print(f"{label_str} — Precision: {precision_all[i]:.4f}, Recall: {recall_all[i]:.4f}, F1: {f1_all[i]:.4f}, Support: {support[i]}")


# In[ ]:





# In[30]:


# LSTM IMPLEMENTATION


# In[31]:


from torch.utils.data import Dataset, DataLoader

class IndexedTweetDataset(Dataset):
    def __init__(self, df):
        self.X = df['padded'].tolist()
        self.y = df['label'].tolist()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

# Create datasets and loaders
train_dataset = IndexedTweetDataset(train_df)
dev_dataset = IndexedTweetDataset(dev_df)
test_dataset = IndexedTweetDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# In[32]:


import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=256):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word2idx['PAD'])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # output_dim = 1 for binary

    def forward(self, x):
        embedded = self.embedding(x)           # [batch_size, seq_len] → [batch_size, seq_len, embed_dim]
        _, (hidden, _) = self.lstm(embedded)   # hidden: [1, batch_size, hidden_dim]
        output = self.fc(hidden[-1])           # Take the last hidden state → [batch_size, 1]
        return output.squeeze(1)               # Final output: [batch_size]


# In[33]:


model = LSTMClassifier(
    vocab_size=len(word2idx),
    embed_dim=100,
    hidden_dim=256
).to(device)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[34]:


from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm

EPOCHS = 10
train_losses = []
dev_accuracies = []
best_dev_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Eval on Dev
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dev_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    dev_acc = accuracy_score(all_labels, all_preds)
    dev_accuracies.append(dev_acc)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Dev Acc={dev_acc:.4f}")

    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        torch.save(model.state_dict(), "Binary_best_lstm_model.pt")
        print(" Best model saved.")


# In[35]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o')
plt.title("Train Loss per Epoch")

plt.subplot(1, 2, 2)
plt.plot(dev_accuracies, marker='o', color='green')
plt.title("Dev Accuracy per Epoch")
plt.show()


# In[36]:


from sklearn.metrics import classification_report

model.load_state_dict(torch.load("Binary_best_lstm_model.pt"))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n Final Evaluation on Test Set:\n")
print(classification_report(all_labels, all_preds, target_names=["negative", "positive"]))





