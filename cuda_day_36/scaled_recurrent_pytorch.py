import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist

# ------------------------
# Download the dataset if needed.
if not os.path.exists("input.txt"):
    os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

# Load the text.
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Build a character-level vocabulary.
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Unique characters:", vocab_size)

# Create mappings from characters to indices and vice versa.
char_to_idx = { ch: i for i, ch in enumerate(chars) }
idx_to_char = { i: ch for i, ch in enumerate(chars) }

def encode(s):
    return [char_to_idx[c] for c in s]

def decode(indices):
    return "".join([idx_to_char[i] for i in indices])

data = torch.tensor(encode(text), dtype=torch.long)

# ------------------------
# Helper function: sample r (number of recurrent iterations)
def sample_r(r_bar, sigma=0.5):
    normal = dist.Normal(torch.log(torch.tensor(r_bar)) - 0.5 * sigma**2, sigma)
    tau = normal.sample()
    lam = torch.exp(tau)
    poisson = dist.Poisson(lam)
    r = poisson.sample().int() + 1
    return r.item()

# ------------------------
# Model Architecture with further reduced hyperparameters.

class PreludeBlock(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class RecurrentBlock(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.adapter = nn.Linear(2 * embed_dim, embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, e, s):
        x = torch.cat([s, e], dim=-1)
        x = self.adapter(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class CodaBlock(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, vocab_size, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.proj(x)
        return logits

class LatentRecurrentTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, lP, lR, lC, num_heads, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(max_seq_len, embed_dim))
        self.prelude = PreludeBlock(lP, embed_dim, num_heads, dropout)
        self.recurrent = RecurrentBlock(lR, embed_dim, num_heads, dropout)
        self.coda = CodaBlock(lC, embed_dim, num_heads, vocab_size, dropout)
        self.embed_dim = embed_dim
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        
    def forward(self, x, r_iters=None):
        batch_size, seq_len = x.shape
        x_emb = self.embedding(x) + self.pos_embedding[:seq_len, :]
        x_emb = x_emb.transpose(0, 1)
        e = self.prelude(x_emb)
        sigma = 1.0
        s = torch.randn_like(e) * sigma
        if r_iters is None:
            r_iters = sample_r(r_bar=33, sigma=0.5)
        for _ in range(r_iters):
            s = self.recurrent(e, s)
        logits = self.coda(s)
        logits = logits.transpose(0, 1)
        return logits

# ------------------------
# Training Setup with further reduced hyperparameters.
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 8      # Reduced batch size
block_size = 32     # Reduced sequence length
num_epochs = 20
learning_rate = 3e-4

# Further reduced model size.
embed_dim = 32      # Reduced embedding dimension
lP = 1              # One prelude layer
lR = 1              # One recurrent layer
lC = 1              # One coda layer
num_heads = 1       # One attention head
max_seq_len = 256   # Reduced maximum sequence length

model = LatentRecurrentTransformer(vocab_size, embed_dim, lP, lR, lC, num_heads, max_seq_len)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x_batch = torch.stack([data[i:i+block_size] for i in ix])
    y_batch = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x_batch.to(device), y_batch.to(device)

def generate_text(model, prompt, generation_length=100, temperature=1.0, device="cpu"):
    model.eval()
    input_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    generated = input_ids[0].tolist()
    with torch.no_grad():
        for _ in range(generation_length):
            logits = model(input_ids)
            last_logits = logits[0, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)
    return decode(generated)

# ------------------------
# Continuous Training Loop.
prompt = "To be, or not to be: that is the question:\n"
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    iterations = 100  # Reduced iterations for demo purposes
    for it in range(iterations):
        optimizer.zero_grad()
        x_batch, y_batch = get_batch(data, batch_size, block_size, device)
        logits = model(x_batch)
        loss = criterion(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (it + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Iteration {it+1}/{iterations}, Loss: {loss.item():.4f}")
    avg_loss = total_loss / iterations
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    sample_text = generate_text(model, prompt, generation_length=100, temperature=1.0, device=device)
    print(f"\n--- Generated Text at Epoch {epoch+1} ---")
    print(sample_text)
    
    torch.save(model.state_dict(), f"latent_recurrent_epoch{epoch+1}.pt")
    
    # Free up cached GPU memory.
    if device == "cuda":
        torch.cuda.empty_cache()
