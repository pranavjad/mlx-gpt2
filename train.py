import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils
import numpy as np
import math

### hyper params
# model
ctx_len = 128
n_emb = 128
dropout = 0.1
head_size = 128
n_heads = 4 
n_layers = 3

# training
num_epochs=20
batch_size=64
lr = 1e-3


### Tokenization
with open('./input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
itos = {i:c for i,c in enumerate(vocab)} # int to string
stoi = {c:i for i,c in enumerate(vocab)} # string to int
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[i] for i in x])
data = encode(text)
split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

### Data Prep
ctx_len = 8
X_train = mx.array([train_data[i:i+ctx_len] for i in range(0, len(train_data) - ctx_len, ctx_len)])
y_train = mx.array([train_data[i+1:i+ctx_len+1] for i in range(0, len(train_data) - ctx_len, ctx_len)])
X_val = mx.array([val_data[i:i+ctx_len] for i in range(0, len(val_data) - ctx_len, ctx_len)])
y_val = mx.array([val_data[i+1:i+ctx_len+1] for i in range(0, len(val_data) - ctx_len, ctx_len)])


def get_batches(X, y, b_size, shuffle=True):
    if shuffle:
        ix = np.arange(X.shape[0])
        np.random.shuffle(ix)
        ix = mx.array(ix)
        X = X[ix]
        y = y[ix]
    for i in range(0, X.shape[0], b_size):
        input = X[i:i+b_size]
        label = y[i:i+b_size]
        yield input, label

### Model Definition
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_emb)
        self.wpe = nn.Embedding(ctx_len, n_emb)
        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_layers)],
        )
        self.ln_f = nn.LayerNorm(dims=n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
        self._init_parameters()
    def __call__(self, x):
        B, T = x.shape
        tok_emb = self.wte(x)
        pos_emb = self.wpe(mx.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    def generate(self, max_new_tokens):
        ctx = mx.zeros((1, 1), dtype=mx.int32)
        for _ in range(max_new_tokens):
          logits = self(ctx[:, -ctx_len:])
          logits = logits[:, -1, :]
          next_tok = mx.random.categorical(logits, num_samples=1)
          ctx = mx.concatenate((ctx, next_tok), axis=1)
        return ctx
    def _init_parameters(self):
        normal_init = nn.init.normal(mean=0.0, std=0.02)
        residual_init = nn.init.normal(mean=0.0, std=(0.02 / math.sqrt(2 * n_layers)))
        new_params = []
        for name, module in self.named_modules():
            if isinstance(module, nn.layers.linear.Linear):
                if 'c_proj' in name:
                    new_params.append((name + '.weight', residual_init(module.weight)))
                else:
                    new_params.append((name + '.weight', normal_init(module.weight)))
                if 'bias' in module:
                    new_params.append((name + '.bias', mx.zeros(module.bias.shape)))
            elif isinstance(module, nn.layers.embedding.Embedding):
                new_params.append((name + '.weight', normal_init(module.weight)))
        self = self.update(utils.tree_unflatten(new_params))

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(n_emb, head_size, bias=False)
        self.q_proj = nn.Linear(n_emb, head_size, bias=False)
        self.v_proj = nn.Linear(n_emb, head_size, bias=False)
        indices = mx.arange(ctx_len)
        mask = indices[:, None] < indices[None]
        self._causal_mask = mask * -1e9
        self.c_proj = nn.Linear(head_size, n_emb)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    def __call__(self, x):
        B, T, C = x.shape
        K = self.k_proj(x)
        Q = self.q_proj(x)
        V = self.v_proj(x)
        mha_shape = (B, T, n_heads, head_size//n_heads)
        K = mx.as_strided(K, (mha_shape)).transpose([0, 2, 1, 3])
        Q = mx.as_strided(Q, (mha_shape)).transpose([0, 2, 1, 3])
        V = mx.as_strided(V, (mha_shape)).transpose([0, 2, 1, 3])
        attn_weights = (Q @ K.transpose([0, 1, 3, 2])) / math.sqrt(Q.shape[-1])
        attn_weights = attn_weights + self._causal_mask[:T, :T]
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)
        o = (attn_weights @ V)
        o = o.transpose([0, 2, 1, 3]).reshape((B, T, head_size))
        o = self.c_proj(self.resid_dropout(o))
        return o

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_emb, 4 * n_emb)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)
    def __call__(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.mha = MultiHeadAttention()
        self.ln_1 = nn.LayerNorm(dims=n_emb)
        self.ln_2 = nn.LayerNorm(dims=n_emb)
    def __call__(self, x):
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

### Training
def loss_fn(model, x, y):
    logits = model(x)
    B, T, C = logits.shape
    logits = logits.reshape(B*T, C)
    y = y.reshape(B*T)
    loss = nn.losses.cross_entropy(logits, y, reduction='mean')
    return loss

model = GPT()
mx.eval(model.parameters())
loss_and_grad = nn.value_and_grad(model, loss_fn)
optimizer = optim.AdamW(learning_rate=lr)

for epoch in range(num_epochs):
    model.train(True)
    running_loss = 0
    batch_cnt = 0
    for input, label in get_batches(X_train, y_train, batch_size):
        batch_cnt += 1
        loss, grads = loss_and_grad(model, input, label)
        optimizer.update(model, grads)
        running_loss += loss.item()
        # compute new parameters and optimizer state
        mx.eval(model.parameters(), optimizer.state)
    avg_train_loss = running_loss / batch_cnt
    model.train(False) # set eval mode
    running_loss = 0
    batch_cnt = 0
    for input, label in get_batches(X_val, y_val, batch_size):
        batch_cnt += 1
        loss = loss_fn(model, input, label)
        running_loss += loss.item()
    avg_val_loss = running_loss / batch_cnt
    print(f"Epoch {epoch:2} | train = {avg_train_loss:.4f} | val = {avg_val_loss:.4f}")

### Inference
completion = decode(model.generate(1000)[0].tolist())
print(completion)
with open('completions.txt', 'w') as f:
    f.write(completion)