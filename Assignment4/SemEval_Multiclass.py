#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# SemEval (multiclass) dataset


# In[1]:


import torch
print(torch.__version__)
print(torch.cuda.is_available())


# In[2]:


import pandas as pd
import spacy
import numpy as np


# In[3]:


from collections import Counter
import re
nlp = spacy.load("en_core_web_sm")


# In[4]:


import pandas as pd

# Load datasets
train_df = pd.read_csv("semeval-2017-train.csv", sep='\t', encoding='utf-8')
dev_df = pd.read_csv("semeval-2017-dev.csv", sep='\t', encoding='utf-8')
test_df = pd.read_csv("semeval-2017-test.csv", sep='\t', encoding='utf-8')

train_df.head()


# In[5]:


dev_df.shape


# In[6]:


for col in test_df.columns:
    print(col)


# In[7]:


print(train_df['label'].value_counts())
#label_map = {-1: "negative", 0: "neutral", 1: "positive"}


# In[8]:


# Basic cleaning
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", '', text)
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


# In[9]:


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

# Stats
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


# In[10]:


# Show vocab stats
print("\n--- Vocabulary Stats ---")
print("Total unique tokens in training data:", len(token_freq))
print("Words kept (freq >= 5):", len(vocab) - 2)
print("Words dropped (replaced with UNK):", len(token_freq) - (len(vocab) - 2))


# In[11]:


print("\n--- Sample Rows from Train Data ---")
for i in range(3):
    print(f"\nOriginal Text: {train_df['text'].iloc[i]}")
    print(f"Tokens: {train_df['tokens'].iloc[i]}")
    print(f"Encoded Tokens: {train_df['encoded_tokens'].iloc[i]}")
    print(f"Padded: {train_df['padded'].iloc[i]}")
    print(f"One-hot Shape: {train_df['onehot'].iloc[i].shape}")


# In[12]:


label_map = {-1: "negative", 0: "neutral", 1: "positive"}
train_df["label_name"] = train_df["label"].map(label_map)

print("\n--- Label Distribution in Train ---")
print(train_df["label_name"].value_counts())

print("\n--- Label Distribution in Dev ---")
dev_df["label_name"] = dev_df["label"].map(label_map)
print(dev_df["label_name"].value_counts())

print("\n--- Label Distribution in Test ---")
test_df["label_name"] = test_df["label"].map(label_map)
print(test_df["label_name"].value_counts())


# In[13]:


import numpy as np
np.set_printoptions(threshold=100, linewidth=160)  # Avoid cutting off output

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


# In[14]:


unk_count = sum([1 for token in train_df['encoded_tokens'].explode() if token == word2idx['UNK']])
print("Total UNK tokens in training data:", unk_count)


# In[15]:


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


# In[16]:


import torch.nn as nn
import torch.nn.functional as F

class FeedForwardSentiment(nn.Module):
    def __init__(self, input_dim, hidden1=256, hidden2=128, output_dim=3):
        super(FeedForwardSentiment, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.output = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        x = self.flatten(x)                   # [batch_size, 21, vocab_size] → [batch_size, 21*vocab_size]
        x = F.relu(self.fc1(x))               # Hidden layer 1 + ReLU
        x = F.relu(self.fc2(x))               # Hidden layer 2 + ReLU
        x = self.output(x)                    # Output logits (for softmax)
        return x
vocab_size = len(word2idx)
input_dim = max_len * vocab_size  # 21 × 10634
output_dim = 3  # multiclass: positive, neutral, negative

model = FeedForwardSentiment(input_dim=input_dim, output_dim=output_dim)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# In[17]:


from torch.utils.data import Dataset, DataLoader
import torch

class LazyTweetDataset(Dataset):
    def __init__(self, df, word2idx, vocab_size, max_len):
        self.df = df
        self.word2idx = word2idx
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_idx = word2idx["PAD"]
        self.label = df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        padded = self.df["padded"].iloc[idx]
        one_hot = np.zeros((self.max_len, self.vocab_size), dtype=np.float32)
        for i, token_idx in enumerate(padded):
            one_hot[i, token_idx] = 1.0
        return torch.tensor(one_hot), torch.tensor(self.label[idx], dtype=torch.long)

train_dataset = LazyTweetDataset(train_df, word2idx, len(word2idx), max_len)
dev_dataset = LazyTweetDataset(dev_df, word2idx, len(word2idx), max_len)
test_dataset = LazyTweetDataset(test_df, word2idx, len(word2idx), max_len)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# In[18]:


label_map = {-1: 0, 0: 1, 1: 2}
train_df["label"] = train_df["label"].map(label_map)
dev_df["label"] = dev_df["label"].map(label_map)
test_df["label"] = test_df["label"].map(label_map)


# In[19]:


print("Train label range:", train_df["label"].unique())
print("Dev label range:", dev_df["label"].unique())
print("Test label range:", test_df["label"].unique())


# In[20]:


import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
EPOCHS = 10
best_dev_acc = 0.0
train_losses = []
dev_accuracies = []
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)

from tqdm.notebook import tqdm  

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
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    dev_acc = accuracy_score(all_labels, all_preds)
    dev_accuracies.append(dev_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {avg_train_loss:.4f} — Dev Accuracy: {dev_acc:.4f}")

    # Save best model
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        torch.save(model.state_dict(), "best_ffnn_model.pt")
        print(" Best model saved!")


# In[21]:


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


# In[22]:


# Load best model
model.load_state_dict(torch.load("best_ffnn_model.pt"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Overall metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

print("\n Final Test Set Evaluation:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Per-label metrics
labels_map = {0: "neutral", 1: "positive", 2: "negative"}
precision_all, recall_all, f1_all, support = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1, 2])

print("\n Per-class Metrics:")
for i, label in enumerate([0, 1, 2]):
    print(f"{labels_map[label]} — Precision: {precision_all[i]:.4f}, Recall: {recall_all[i]:.4f}, F1: {f1_all[i]:.4f}, Support: {support[i]}")


# In[ ]:


#LSTM IMPLEMENTATION


# In[23]:


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


# In[24]:


import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=256, output_dim=3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word2idx['PAD'])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)                 # [B, L] → [B, L, E]
        _, (hidden, _) = self.lstm(embedded)         # Get final hidden state
        return self.fc(hidden[-1])                   # [B, H] → [B, output_dim]


# In[25]:


model = LSTMClassifier(
    vocab_size=len(word2idx),
    embed_dim=100,            # You can tune this
    hidden_dim=256,
    output_dim=3              # 3-way sentiment
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[26]:


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
        inputs, labels = inputs.to(device), labels.to(device)

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
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    dev_acc = accuracy_score(all_labels, all_preds)
    dev_accuracies.append(dev_acc)

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Dev Acc={dev_acc:.4f}")

    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        torch.save(model.state_dict(), "best_lstm_model.pt")
        print(" Best model saved.")


# In[27]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o')
plt.title("Train Loss per Epoch")

plt.subplot(1, 2, 2)
plt.plot(dev_accuracies, marker='o', color='green')
plt.title("Dev Accuracy per Epoch")
plt.show()


# In[28]:


from sklearn.metrics import classification_report

model.load_state_dict(torch.load("best_lstm_model.pt"))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n Final Evaluation on Test Set:\n")
print(classification_report(all_labels, all_preds, target_names=["neutral", "positive", "negative"]))


# In[ ]:




