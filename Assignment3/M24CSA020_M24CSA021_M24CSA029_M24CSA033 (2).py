# CODE 1: LSTM IMPLEMENTATION


#!/usr/bin/env python
# coding: utf-8

# LSTM Implementation 

# In[1]:


import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # Server with CUDA GPU
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Mac M1/M2 GPU
else:
    DEVICE = torch.device("cpu")  # Fallback CPU

print(f"Using device: {DEVICE}")


# In[2]:


eng_train_file = 'english.train'
hin_train_file = 'hindi.train'
eng_test_file = 'english.test'
hin_test_file = 'hindi.test'


# In[3]:


def load_data(eng_file, hin_file):
    with open(eng_file, 'r', encoding='utf-8') as ef, open(hin_file, 'r', encoding='utf-8') as hf:
        eng_sentences = [line.strip().lower().split() for line in ef]
        hin_sentences = [line.strip().lower().split() for line in hf]
    return eng_sentences, hin_sentences

eng_train, hin_train = load_data(eng_train_file, hin_train_file)
eng_test, hin_test = load_data(eng_test_file, hin_test_file)

# Add <SOS> and <EOS> to both source and target sentences
eng_train = [['<SOS>'] + s + ['<EOS>'] for s in eng_train]
hin_train = [['<SOS>'] + s + ['<EOS>'] for s in hin_train]

eng_test  = [['<SOS>'] + s + ['<EOS>'] for s in eng_test]
hin_test  = [['<SOS>'] + s + ['<EOS>'] for s in hin_test]

print(f"Sample English Sentence: {eng_train[0]}")
print(f"Sample Hindi Sentence: {hin_train[0]}")


# In[4]:


def build_vocab(sentences):
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

eng_vocab = build_vocab(eng_train + eng_test)
hin_vocab = build_vocab(hin_train + hin_test)

print(f"English Vocab Size: {len(eng_vocab)}")
print(f"Hindi Vocab Size: {len(hin_vocab)}")

# Sample display: get any word in vocab
eng_inv_vocab = list(eng_vocab.items())
hin_inv_vocab = list(hin_vocab.items())

print(f"Sample English Word Mapping: {eng_inv_vocab[5]}")
print(f"Sample Hindi Word Mapping: {hin_inv_vocab[5]}")




# In[9]:


# Load the precomputed embedding matrices
eng_emb_matrix = np.load('eng_embedding_matrix.npy')
hin_emb_matrix = np.load('hin_embedding_matrix.npy')


# In[10]:


def merge_and_split(eng_train, eng_test, hin_train, hin_test, seed=42):
    assert len(eng_train) == len(hin_train)
    assert len(eng_test) == len(hin_test)

    # Combine and zip for alignment
    all_eng = eng_train + eng_test
    all_hin = hin_train + hin_test
    combined = list(zip(all_eng, all_hin))

    # Shuffle with fixed seed
    random.seed(seed)
    random.shuffle(combined)

    # Compute splits
    total = len(combined)
    train_end = int(0.7 * total)
    val_end = train_end + int(0.15 * total)

    train = combined[:train_end]
    val   = combined[train_end:val_end]
    test  = combined[val_end:]

    # Unzip
    eng_train, hin_train = zip(*train)
    eng_val, hin_val     = zip(*val)
    eng_test, hin_test   = zip(*test)

    return list(eng_train), list(eng_val), list(eng_test), list(hin_train), list(hin_val), list(hin_test)


# In[11]:


eng_train, eng_val, eng_test, hin_train, hin_val, hin_test = merge_and_split(
    eng_train, eng_test, hin_train, hin_test, seed=42
)


# In[31]:


print(f"Train size: {len(eng_train)}")
print(f"Val size:   {len(eng_val)}")
print(f"Test size:  {len(eng_test)}")

print("Sample aligned pair:")
print("ENG:", eng_train[2001])
print("HIN:", hin_train[2001])


# In[32]:


# Constants
EMBEDDING_DIM = 300
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Converts token list to indices
def sentence_to_indices(tokens, vocab):
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]

# Dataset
class TranslationDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, source_vocab, target_vocab):
        self.source = [sentence_to_indices(s, source_vocab) for s in source_sentences]
        self.target = [sentence_to_indices(s, target_vocab) for s in target_sentences]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return torch.tensor(self.source[idx]), torch.tensor(self.target[idx])

# Collate for padding
def collate_fn(batch):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, padding_value=0)
    tgt = nn.utils.rnn.pad_sequence(tgt, padding_value=0)
    return src, tgt


# In[33]:


# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_matrix, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(emb_matrix, dtype=torch.float32), freeze=False)
        self.lstm = nn.LSTM(EMBEDDING_DIM, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_matrix, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(emb_matrix, dtype=torch.float32), freeze=False)
        self.lstm = nn.LSTM(EMBEDDING_DIM, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)  # shape: [1, batch_size]
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        batch_size = tgt.shape[1]
        max_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(max_len, batch_size, tgt_vocab_size).to(DEVICE)

        hidden, cell = self.encoder(src)

        input = tgt[0, :]  # <sos> tokens

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            input = tgt[t]  # Teacher forcing

        return outputs


# In[34]:


BATCH_SIZE = 64

train_dataset = TranslationDataset(eng_train, hin_train, eng_vocab, hin_vocab)
val_dataset   = TranslationDataset(eng_val, hin_val, eng_vocab, hin_vocab)
test_dataset  = TranslationDataset(eng_test, hin_test, eng_vocab, hin_vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)



# In[35]:


# Grab a single batch from train_loader
src_batch, tgt_batch = next(iter(train_loader))

print("Source batch shape:", src_batch.shape)
print("Target batch shape:", tgt_batch.shape)


# In[36]:


# Model setup
INPUT_DIM = len(eng_vocab)
OUTPUT_DIM = len(hin_vocab)
HIDDEN_DIM = 512

encoder = Encoder(INPUT_DIM, eng_emb_matrix, HIDDEN_DIM).to(DEVICE)
decoder = Decoder(OUTPUT_DIM, hin_emb_matrix, HIDDEN_DIM).to(DEVICE)
model = Seq2Seq(encoder, decoder).to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=hin_vocab[PAD_TOKEN])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for src, tgt in tqdm(loader, desc="Training", leave=False):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()

        output = model(src, tgt)
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        tgt = tgt[1:].view(-1)

        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)

# Evaluation function
def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            output = model(src, tgt)
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            tgt = tgt[1:].view(-1)

            loss = criterion(output, tgt)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


# In[18]:


NUM_EPOCHS = 30
best_val_loss = float('inf')
patience = 10
wait = 0
start_epoch = 0

'''
if os.path.exists('checkpoint.pth'):
    checkpoint = torch.load('checkpoint.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_val_loss = checkpoint['best_val_loss']
    start_epoch = checkpoint['epoch'] + 1
    print(f" Resuming training from epoch {start_epoch}")
'''

for epoch in range(start_epoch, NUM_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, 'checkpoint.pth')
        print("✔ Checkpoint saved (new best).")

    else:
        wait += 1
        print(f" No improvement. Patience: {wait}/{patience}")

        if wait >= patience:
            print(" Early stopping triggered.")
            break


# In[19]:


# # Run this if training is interrupted in between
# checkpoint = torch.load('checkpoint.pth', map_location=DEVICE)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to(DEVICE)
# model.eval()


# In[20]:


def translate(model, src_sentence, eng_vocab, hin_vocab, hin_inv_vocab, max_len=50):
    model.eval()
    
    # Convert source sentence (tokens) to indices
    src_indices = [eng_vocab.get(SOS_TOKEN, 1)] + \
                  [eng_vocab.get(token, eng_vocab[UNK_TOKEN]) for token in src_sentence] + \
                  [eng_vocab.get(EOS_TOKEN, 2)]

    src_tensor = torch.tensor(src_indices).unsqueeze(1).to(DEVICE)  # shape: [src_len, 1]

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    tgt_indices = [hin_vocab[SOS_TOKEN]]
    outputs = []

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_indices[-1]]).to(DEVICE)
        with torch.no_grad():
            output, hidden, cell = model.decoder(tgt_tensor, hidden, cell)
        
        pred_token = output.argmax(1).item()
        
        if pred_token == hin_vocab[EOS_TOKEN]:
            break

        outputs.append(pred_token)
        tgt_indices.append(pred_token)

    # Convert indices back to tokens
    translated_words = [hin_inv_vocab[idx] for idx in outputs if idx in hin_inv_vocab]
    return translated_words


# In[30]:


import random
hin_inv_vocab = {idx: word for word, idx in hin_vocab.items()}

# Pick a random index
rand_idx = random.randint(0, len(eng_test) - 1)
test_sentence = eng_test[rand_idx]

clean_src = test_sentence[1:-1]  # removes first and last token

# Translate
translation = translate(model, test_sentence, eng_vocab, hin_vocab, hin_inv_vocab)

# Display
print("Random Test Index:", rand_idx)
print("English:", " ".join(clean_src))
print("Hindi:", " ".join(translation))


# In[22]:


from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
smoothie = SmoothingFunction().method4

def evaluate_bleu(model, test_data, references, eng_vocab, hin_vocab, hin_inv_vocab):
    all_preds = []
    all_refs = []

    for i, src_sentence in enumerate(tqdm(test_data, desc="Translating")):
        pred = translate(model, src_sentence, eng_vocab, hin_vocab, hin_inv_vocab)
        ref = references[i]  # already tokenized and lowercased

        all_preds.append(pred)
        all_refs.append([ref])  # wrap reference in list for corpus_bleu

    bleu = corpus_bleu(all_refs, all_preds, smoothing_function=smoothie)
    return bleu

# Create inverse vocab once
hin_inv_vocab = {idx: word for word, idx in hin_vocab.items()}

# Compute BLEU
bleu_score = evaluate_bleu(model, eng_test, hin_test, eng_vocab, hin_vocab, hin_inv_vocab)
print(f"\nCorpus BLEU Score: {bleu_score:.4f}")


# In[ ]:


############################################################################################################
############################################################################################################
############################################################################################################


#CODE 2 : TRANSFORMER IMPLEMENTATION ( Better Results )


#!/usr/bin/env python
# coding: utf-8

# Transformer Implementation 

# In[2]:


import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
import os
import math

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # Server with CUDA GPU
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Mac M1/M2 GPU
else:
    DEVICE = torch.device("cpu")  # Fallback CPU

print(f"Using device: {DEVICE}")


# In[3]:


eng_train_file = 'english.train'
hin_train_file = 'hindi.train'
eng_test_file = 'english.test'
hin_test_file = 'hindi.test'


# In[4]:


def load_data(eng_file, hin_file):
    with open(eng_file, 'r', encoding='utf-8') as ef, open(hin_file, 'r', encoding='utf-8') as hf:
        eng_sentences = [line.strip().lower().split() for line in ef]
        hin_sentences = [line.strip().lower().split() for line in hf]
    return eng_sentences, hin_sentences

eng_train, hin_train = load_data(eng_train_file, hin_train_file)
eng_test, hin_test = load_data(eng_test_file, hin_test_file)

# Add <SOS> and <EOS> to both source and target sentences
eng_train = [['<SOS>'] + s + ['<EOS>'] for s in eng_train]
hin_train = [['<SOS>'] + s + ['<EOS>'] for s in hin_train]

eng_test  = [['<SOS>'] + s + ['<EOS>'] for s in eng_test]
hin_test  = [['<SOS>'] + s + ['<EOS>'] for s in hin_test]

print(f"Sample English Sentence: {eng_train[0]}")
print(f"Sample Hindi Sentence: {hin_train[0]}")


# In[5]:


def build_vocab(sentences):
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for sentence in sentences:
        for word in sentence:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

eng_vocab = build_vocab(eng_train + eng_test)
hin_vocab = build_vocab(hin_train + hin_test)

print(f"English Vocab Size: {len(eng_vocab)}")
print(f"Hindi Vocab Size: {len(hin_vocab)}")

# Sample display: get any word in vocab
eng_inv_vocab = list(eng_vocab.items())
hin_inv_vocab = list(hin_vocab.items())

print(f"Sample English Word Mapping: {eng_inv_vocab[5]}")
print(f"Sample Hindi Word Mapping: {hin_inv_vocab[5]}")


# In[6]:


# In[10]:


# Load the precomputed embedding matrices
eng_emb_matrix = np.load('eng_embedding_matrix.npy')
hin_emb_matrix = np.load('hin_embedding_matrix.npy')


# In[11]:


def merge_and_split(eng_train, eng_test, hin_train, hin_test, seed=42):
    assert len(eng_train) == len(hin_train)
    assert len(eng_test) == len(hin_test)

    # Combine and zip for alignment
    all_eng = eng_train + eng_test
    all_hin = hin_train + hin_test
    combined = list(zip(all_eng, all_hin))

    # Shuffle with fixed seed
    random.seed(seed)
    random.shuffle(combined)

    # Compute splits
    total = len(combined)
    train_end = int(0.7 * total)
    val_end = train_end + int(0.15 * total)

    train = combined[:train_end]
    val   = combined[train_end:val_end]
    test  = combined[val_end:]

    # Unzip
    eng_train, hin_train = zip(*train)
    eng_val, hin_val     = zip(*val)
    eng_test, hin_test   = zip(*test)

    return list(eng_train), list(eng_val), list(eng_test), list(hin_train), list(hin_val), list(hin_test)


# In[12]:


eng_train, eng_val, eng_test, hin_train, hin_val, hin_test = merge_and_split(
    eng_train, eng_test, hin_train, hin_test, seed=42
)


# In[13]:


print(f"Train size: {len(eng_train)}")
print(f"Val size:   {len(eng_val)}")
print(f"Test size:  {len(eng_test)}")

print("Sample aligned pair:")
print("ENG:", eng_train[2000])
print("HIN:", hin_train[2000])


# In[14]:


# Constants
EMBEDDING_DIM = 300
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Converts token list to indices
def sentence_to_indices(tokens, vocab):
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]

# Dataset
class TranslationDataset(Dataset):
    def __init__(self, source_sentences, target_sentences, source_vocab, target_vocab):
        self.source = [sentence_to_indices(s, source_vocab) for s in source_sentences]
        self.target = [sentence_to_indices(s, target_vocab) for s in target_sentences]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return torch.tensor(self.source[idx]), torch.tensor(self.target[idx])

def collate_fn(batch):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    return src, tgt


# In[15]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, _ = scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)

        return self.out(attn_output)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_output, enc_output, memory_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, ff_dim, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, ff_dim, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, memory_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_layers=4,
                 num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, ff_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, ff_dim, dropout)
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)

    def make_pad_mask(self, seq, pad_idx=0):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_look_ahead_mask(self, size):
        return torch.tril(torch.ones(size, size)).type(torch.bool)

    def forward(self, src, tgt):
        src_mask = self.make_pad_mask(src)
        tgt_mask = self.make_pad_mask(tgt) & self.make_look_ahead_mask(tgt.size(1)).to(tgt.device)

        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return self.final_linear(dec_out)


# In[16]:


# Hyperparameters
D_MODEL = 128
NUM_LAYERS = 4
NUM_HEADS = 8
FF_DIM = 512
DROPOUT = 0.1
BATCH_SIZE = 64
EPOCHS=30

SRC_VOCAB_SIZE = len(eng_vocab)
TGT_VOCAB_SIZE = len(hin_vocab)

model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    dropout=DROPOUT
).to(DEVICE)

PAD_IDX = eng_vocab[PAD_TOKEN]

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_dataset = TranslationDataset(eng_train, hin_train, eng_vocab, hin_vocab)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


# In[17]:


from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

def train(model, data_loader, optimizer, criterion, device, epochs):
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        total_tokens = 0
        correct_tokens = 0

        print(f"\n Epoch {epoch+1}/{epochs}")
        progress_bar = tqdm(data_loader, desc=f"Training", leave=False)

        for src, tgt in progress_bar:
            src = src.to(device)  # (B, src_len)
            tgt = tgt.to(device)  # (B, tgt_len)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)  # (B, tgt_len-1, vocab_size)
            output = output.reshape(-1, output.size(-1))  # (B*(tgt_len-1), vocab_size)
            tgt_output = tgt_output.reshape(-1)  # (B*(tgt_len-1))

            loss = criterion(output, tgt_output)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Accuracy calc
            pred = output.argmax(dim=1)
            mask = tgt_output != PAD_IDX
            correct_tokens += (pred == tgt_output).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(data_loader)
        acc = 100 * correct_tokens / total_tokens
        print(f" Epoch {epoch+1} completed — Avg Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")


# In[18]:


train(model, train_loader, optimizer, criterion, DEVICE, epochs=EPOCHS)


# In[19]:


def greedy_decode(model, src_sentence, src_vocab, tgt_vocab, max_len=50):
    model.eval()

    src_tokens = sentence_to_indices(src_sentence, src_vocab)
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(DEVICE)
    tgt_tensor = torch.tensor([[tgt_vocab[SOS_TOKEN]]], dtype=torch.long).to(DEVICE)

    inv_tgt_vocab = {idx: token for token, idx in tgt_vocab.items()}

    with torch.no_grad():
        for _ in range(max_len):
            output = model(src_tensor, tgt_tensor)  # (1, len, vocab)
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)  # shape: (1, 1)
            tgt_tensor = torch.cat((tgt_tensor, next_token), dim=1)

            if next_token.item() == tgt_vocab[EOS_TOKEN]:
                break

    decoded = [inv_tgt_vocab.get(tok.item(), UNK_TOKEN)
               for tok in tgt_tensor.squeeze()[1:-1]]  # skip SOS and EOS
    return ' '.join(decoded)


# In[20]:


test_dataset = TranslationDataset(eng_test, hin_test, eng_vocab, hin_vocab)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
def evaluate_on_test(model, test_loader, src_vocab, tgt_vocab):
    model.eval()
    correct = 0
    total = 0
    inv_tgt_vocab = {idx: tok for tok, idx in tgt_vocab.items()}

    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            output = greedy_decode(model, src[0].tolist(), src_vocab, tgt_vocab)
            target_sentence = [inv_tgt_vocab.get(tok.item(), UNK_TOKEN)
                               for tok in tgt[0][1:-1] if tok.item() != PAD_IDX]

            if output.strip() == ' '.join(target_sentence).strip():
                correct += 1
            total += 1

    print(f"Exact Match Accuracy on Test Set: {100 * correct / total:.2f}%")
evaluate_on_test(model, test_loader, eng_vocab, hin_vocab)


# In[26]:


import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from tqdm import tqdm

# def compute_bleu(model, test_loader, src_vocab, tgt_vocab, max_len=50):
#     model.eval()
#     smoothie = SmoothingFunction().method4

#     inv_tgt_vocab = {idx: tok for tok, idx in tgt_vocab.items()}
#     scores = []

#     with torch.no_grad():
#         for src, tgt in tqdm(test_loader, desc="Computing BLEU", total=len(test_loader)):
#             src = src.to(DEVICE)
#             tgt = tgt.to(DEVICE)

#             pred_sentence = greedy_decode(model, src[0].tolist(), src_vocab, tgt_vocab, max_len=max_len)
#             pred_tokens = pred_sentence.strip().split()

#             ref_tokens = [inv_tgt_vocab.get(tok.item(), UNK_TOKEN)
#                           for tok in tgt[0][1:-1] if tok.item() != PAD_IDX]

#             if len(pred_tokens) > 0 and len(ref_tokens) > 0:
#                 score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
#                 scores.append(score)

#     avg_bleu = sum(scores) / len(scores) if scores else 0
#     print(f"\nCorpus BLEU Score: {avg_bleu * 100:.2f}")

# compute_bleu(model, test_loader, eng_vocab, hin_vocab)


# In[ ]:


def compute_bleu(model, test_loader, src_vocab, tgt_vocab, max_len=50, max_samples=100):
    model.eval()
    smoothie = SmoothingFunction().method4
    inv_tgt_vocab = {idx: tok for tok, idx in tgt_vocab.items()}

    scores = []
    total = 0

    with torch.no_grad():
        for i, (src, tgt) in enumerate(tqdm(test_loader, desc="Computing BLEU", total=max_samples)):
            if i >= max_samples:
                break

            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            pred_sentence = greedy_decode(model, src[0].tolist(), src_vocab, tgt_vocab, max_len=max_len)
            pred_tokens = pred_sentence.strip().split()

            ref_tokens = [inv_tgt_vocab.get(tok.item(), UNK_TOKEN)
                          for tok in tgt[0][1:-1] if tok.item() != PAD_IDX]

            if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
                scores.append(score)

            total += 1

    avg_bleu = sum(scores) / len(scores) if scores else 0
    print(f"\nSampled Corpus BLEU Score on {total} examples: {avg_bleu * 100:.2f}")
compute_bleu(model, test_loader, eng_vocab, hin_vocab, max_len=50, max_samples=3000)


# In[24]:


def show_random_prediction(model, eng_test, hin_test, eng_vocab, hin_vocab):
    inv_hin_vocab = {idx: word for word, idx in hin_vocab.items()}

    # Randomly select a test sample
    idx = random.randint(0, len(eng_test) - 1)
    src_tokens = eng_test[idx]
    tgt_tokens = hin_test[idx]

    # Clean out <SOS>/<EOS>
    clean_src = [tok for tok in src_tokens if tok not in ("<SOS>", "<EOS>")]
    clean_tgt = [tok for tok in tgt_tokens if tok not in ("<SOS>", "<EOS>")]

    # Predict translation
    predicted_sentence = greedy_decode(model, clean_src, eng_vocab, hin_vocab)

    # Display
    print("\n Random Test Sample")
    print(" Index:              ", idx)
    print(" English Input:      ", " ".join(clean_src))
    print(" Ground Truth Hindi: ", " ".join(clean_tgt))
    print(" Predicted Hindi:    ", predicted_sentence)
show_random_prediction(model, eng_test, hin_test, eng_vocab, hin_vocab)


