import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
# batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
# max_iters = 5000
eval_interval = 500 #500
# learning_rate = 3e-4
# Since I am on mac i want to use MPS, so i added this check to enable it
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20 # 200


# Reduced the number of layers and heads to make it run faster
# n_embd = 159
# n_head = 3
# n_layer = 3


# Test 1 
# n_layer=4
# n_head=4
# n_embd=256 

# # Test 2 - increase epochs
# n_embd = 320
# n_head = 4
# n_layer = 5
# batch_size = 128
# max_iters = 8000
# dropout = 0.15
# batch_size = 128


# Test 2 - increase epochs
# n_embd = 320
# n_head = 4
# n_layer = 5
# batch_size = 128
# max_iters = 8000

# Test 3 - 
n_embd = 128 #320
n_head = 4
n_layer = 2 #5

batch_size = 32 #128
max_iters = 200 #8000
dropout = 0.15

learning_rate = 1e-4
dropout = 0.15




# dropout = 0.2
# ------------

torch.manual_seed(1337)

# Debug flag: set to False to silence debug prints
DEBUG = True

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# input = 'input_childSpeech_testSet.txt'
# input = 'finalAssignment_musicDataset/inputMelodiesAugmented.txt'
input = 'finalAssignment_musicDataset/inputMelodiesAugmented_updated.txt'

# input = 'finalAssignment_musicDataset/inputMelodiesAugmented_WithTiming.txt'
with open(input, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, top_p=0.9):
        def _top_p_truncate(logits, top_p):
            """Helper to zero-out logits outside the nucleus (top-p) region.

            logits: Tensor of shape (B, V)
            returns: truncated_logits with tokens outside the cumulative top_p set to -inf
            """
            # sort logits descending and compute softmax on sorted values
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            # mark tokens to remove where cumulative probability exceeds top_p
            to_remove = cumulative_probs > top_p
            # keep first token over threshold to avoid empty set
            to_remove[..., 1:] = to_remove[..., :-1].clone()
            to_remove[..., 0] = False

            # set removed logits to -inf
            sorted_logits[to_remove] = float('-inf')

            # scatter back to original ordering
            truncated = torch.full_like(logits, float('-inf'))
            truncated.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            return truncated

        for _ in range(max_new_tokens):
            # compute logits for the next token
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            # truncate logits using nucleus (top-p) rule and sample
            truncated_logits = _top_p_truncate(logits, top_p)
            new_probs = F.softmax(truncated_logits, dim=-1)
            idx_next = torch.multinomial(new_probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
if DEBUG:
    print("DEBUG: model instantiated and moved to device ->", device)
    try:
        print(f"DEBUG: hyperparams -> n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, batch_size={batch_size}, max_iters={max_iters}, eval_interval={eval_interval}")
    except Exception:
        print("DEBUG: hyperparams not all available")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# Early Stopping Parameters
patience = 4  # Number of evaluations to wait for improvement
best_val_loss = float('inf')  # Initialize best validation loss
counter = 0  # Initialize counter

try:
    if DEBUG:
        print(f"DEBUG: Starting training loop (max_iters={max_iters})")
    for iter in range(max_iters):

        if DEBUG and (iter % 10 == 0):
            print(f"DEBUG: iter {iter} start")

        if iter % eval_interval == 0 or iter == max_iters - 1:
            if DEBUG:
                print("DEBUG: Calling estimate_loss()...")
            losses = estimate_loss()
            train_loss = losses['train']
            val_loss = losses['val']
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            if DEBUG:
                print(f"DEBUG: estimate_loss returned -> train={train_loss:.4f}, val={val_loss:.4f}")
            
            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered. Stopping training.")
                    break

        # Sample a batch of data
        xb, yb = get_batch('train')
        if DEBUG:
            try:
                print(f"DEBUG: get_batch returned xb.shape={tuple(xb.shape)} yb.shape={tuple(yb.shape)}")
            except Exception:
                print("DEBUG: get_batch returned shapes (unable to read)")

        # Forward pass
        logits, loss = model(xb, yb)
        if DEBUG:
            try:
                print(f"DEBUG: forward computed loss={loss.item():.6f}")
            except Exception:
                print("DEBUG: forward computed loss (unable to read)")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if DEBUG:
            print(f"DEBUG: optimizer.step() completed for iter {iter}")

except KeyboardInterrupt:
    print("\n----")
    # torch.save(model.state_dict(), 'interrupted_model.pth')
    # print("Model state saved as 'interrupted_model.pth'.")

# for iter in range(max_iters):

#     # every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0 or iter == max_iters - 1:
#         losses = estimate_loss()
#         train_loss = losses['train']
#         val_loss = losses['val']
#         print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        
#         # Check for improvement in validation loss
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss  # Update best validation loss
#             counter = 0  # Reset counter
#         else:
#             counter += 1  # Increment counter
#             if counter >= patience:
#                 print("Early stopping triggered. Stopping training.")
#                 break  # Exit the training loop

#     # sample a batch of data
#     xb, yb = get_batch('train')

#     # evaluate the loss
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
if DEBUG:
    print("DEBUG: Starting generation...")
result = m.generate(context, max_new_tokens=500)
decoded = decode(result[0].tolist())
print(decoded)
if DEBUG:
    print("DEBUG: Generation complete; length=", len(decoded))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))


