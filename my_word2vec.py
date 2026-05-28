import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm

# `Counter` can count word frequencies 
# `defaultdict` creates dictionaries with default values automatically
from collections import Counter, defaultdict


corpus = """
the king loves the queen and the queen loves the king
the man is strong and the woman is wise
the prince will become king and the princess will become queen
dogs and cats are animals
the dog chased the cat across the yard
lions and tigers are wild animals
the lion is the king of animals
paris is the capital of france
berlin is the capital of germany
rome is the capital of italy
france and germany are countries in europe
italy and spain are countries in europe
the president leads the country
the king rules the kingdom
power comes from the people
""".strip().lower().split() #strip removes the white spaces and split converts the text into list of individual words


vocab = sorted(set(corpus))

# below this creates a dictionary , if king is 13th word then word2idx["king"]=12
word2idx = {w:i for i,w in enumerate(vocab)}

# vice-versa - for index back to original word, useful for viewing results after training
idx2word = {i:w for w,i in word2idx.items()}

V = len(vocab)

# words to integer-ids
encoded = [word2idx[w] for w in corpus]


print(f"Vocabulary Size: {V}")
print(f"Corpus Length: {len(corpus)} tokens")
print(f"Sample Vocab: {vocab[:10]}")



# Training Pair Generation

# SkipGram pair function 
# Returns list of (center_idx,context_idx)
def get_skipgram_pairs(encoded, window=2):
    pairs = []

    for i, center in enumerate(encoded):
        start = max(0, i - window)
        end = min(len(encoded), i + window + 1) # because the pointer stops before the end value so we need to add 1 to include it
        for j in range(start,end):
            if j!=i:
                pairs.append((center,encoded[j]))
    return pairs


# Cbow pair function
# Returns list of (context_list, center_idx)
def get_cbow_pairs(encoded, window=2):
    pairs = []
    for i,center in enumerate(encoded):
        start = max(0,i - window)
        end = min(len(encoded), i+window+1)
        context = [encoded[j] for j in range(start,end) if j!=i]
        if context: # checks context list is not empty
            pairs.append((context,center))
    return pairs


# Building the dataset
skipgram_pairs = get_skipgram_pairs(encoded,window=2)
cbow_pairs = get_cbow_pairs(encoded,window=2)

print(f"\nSkip-gram pairs: {len(skipgram_pairs)}")
print(f"Cbow-pairs: {len(cbow_pairs)}")
print(f"Skip-gram sample: center={idx2word[skipgram_pairs[0][0]]!r}, "
      f"context={idx2word[skipgram_pairs[0][1]]!r}")


# Utilities

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def one_hot(idx, size):
    v = np.zeros(size)
    v[idx] = 1.0
    return v


# Skip-Gram Model

class SkipGram:
    """
    - `W_in` is the input embedding matrix,
    - `W_out` is the output weight matrix,
    - the model takes one center word and predicts a distribution over context words. 
    """

    def __init__(self,vocab_size, embed_dim):
        self.V = vocab_size
        self.E = embed_dim

        # Xavier-ish init: small random weights
        self.W_in = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_out = np.random.randn(embed_dim, vocab_size) * 0.01

    def forward(self, center_idx):
        h = self.W_in[center_idx]
        logits = h @ self.W_out
        probs = softmax(logits)
        return h, logits, probs
    
    def backward(self, center_idx, context_idx, h, probs, lr):
        d_logits = probs.copy()
        d_logits[context_idx] -= 1.0

        d_W_out = np.outer(h, d_logits)
        d_h = self.W_out @ d_logits

        self.W_out -= lr * d_W_out
        self.W_in[center_idx] -= lr * d_h

    def train(self, pairs, epochs=200, lr=0.05):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            np.random.shuffle(pairs)
            for center_idx, context_idx in pairs:
                h, logits, probs = self.forward(center_idx)
                loss = -np.log(probs[context_idx] + 1e-9)
                total_loss +=loss
                self.backward(center_idx, context_idx, h, probs, lr)

            avg_loss = total_loss / len(pairs)
            losses.append(avg_loss)
            if(epoch + 1) % 10 == 0:
                print(f"  [SkipGram] Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")
        return losses
    
    @property
    def embeddings(self):
        """Final word vectors = rows of W_in."""
        return self.W_in
    

# Cbow Model

class CBOW:

    def __init__(self, vocab_size, embed_dim):
        self.V = vocab_size
        self.E = embed_dim
        self.W_in = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W_out = np.random.randn(embed_dim, vocab_size) * 0.01

    def forward(self, context_indices):
        h = self.W_in[context_indices].mean(axis=0)
        logits = h @ self.W_out
        probs = softmax(logits)
        return h, probs

    def backward(self, context_indices, center_idx, h, probs, lr):
        d_logits = probs.copy()
        d_logits[center_idx] -= 1.0

        d_W_out = np.outer(h, d_logits)
        d_h = self.W_out @ d_logits

        self.W_out -= lr * d_W_out

        d_context = d_h / len(context_indices)
        for idx in context_indices:
            self.W_in[idx] -= lr * d_context

    def train(self, pairs, epochs=200, lr=0.05):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0
            np.random.shuffle(pairs)
            for context_indices, center_idx in pairs:
                h, probs = self.forward(context_indices)
                loss = -np.log(probs[center_idx] + 1e-9)
                total_loss += loss
                self.backward(context_indices, center_idx, h, probs, lr)

            avg_loss = total_loss / len(pairs)
            losses.append(avg_loss)
            if(epoch + 1) % 10 == 0:
                print(f" [CBOW]     Epoch {epoch+1:3d}  |  Loss {avg_loss:.4f}") 
        return losses        
    
    @property
    def embeddings(self):
        return self.W_in
    

def pairwise_sq_distances(X):
    """Efficient ||x_i - x_j||^2 using the identity (a-b)^2 = a^2 - 2ab + b^2."""
    sum_sq = np.sum(X ** 2, axis=1)
    return sum_sq[:, None] + sum_sq[None, :] - 2 * (X @ X.T)


def high_dim_affinities(X, perplexity=5.0, tol=1e-5, max_iter=50):
    """
    Compute conditional probabilities p_{j|i} in high-dim space.
    Uses binary search to find sigma_i that achieves the target perplexity.
    Perplexity ≈ effective number of neighbors.
    """
    n = X.shape[0]
    D = pairwise_sq_distances(X)
    P = np.zeros((n, n))
    target_entropy = np.log(perplexity)

    for i in range(n):
        d_i   = np.delete(D[i], i)     # distances from point i to all others
        lo, hi = 0.0, 1e6
        beta   = 1.0                   # beta = 1 / (2 * sigma^2)

        for _ in range(max_iter):
            exp_d   = np.exp(-d_i * beta)
            sum_exp = exp_d.sum() + 1e-9
            p_cond  = exp_d / sum_exp
            entropy = -np.sum(p_cond * np.log(p_cond + 1e-9))

            if abs(entropy - target_entropy) < tol:
                break
            if entropy < target_entropy:   # perplexity too low → decrease beta
                hi   = beta
                beta = (lo + hi) / 2
            else:                          # perplexity too high → increase beta
                lo   = beta
                beta = (lo + hi) / 2

        # Insert 0 for self-similarity
        p_row = np.insert(p_cond, i, 0)
        P[i]  = p_row

    # Symmetrize and normalize
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    return P


def tsne(X, n_components=2, perplexity=5.0, lr=200.0, n_iter=1000, seed=42):
    """
    t-SNE: minimize KL(P || Q) where
      P = high-dim Gaussian affinities
      Q = low-dim Student-t affinities (heavy tail prevents crowding)

    Gradient: dC/dY_i = 4 * Σ_j (p_ij - q_ij)(y_i - y_j)(1 + ||y_i-y_j||^2)^{-1}
    """
    np.random.seed(seed)
    n = X.shape[0]

    print(f"\nComputing high-dim affinities (perplexity={perplexity})...")
    P = high_dim_affinities(X, perplexity=perplexity)
    P *= 4.0   # early exaggeration: push clusters apart early on

    # Random low-dim initialization
    Y     = np.random.randn(n, n_components) * 0.01
    Y_old = Y.copy()
    gains = np.ones_like(Y)
    momentum = 0.5

    for it in range(n_iter):
        # Low-dim Student-t affinities
        D_low = pairwise_sq_distances(Y)
        num   = 1.0 / (1.0 + D_low)
        np.fill_diagonal(num, 0)
        Q     = num / (num.sum() + 1e-9)
        Q     = np.maximum(Q, 1e-12)

        # Gradient
        PQ    = P - Q                               # shape (n, n)
        dY    = np.zeros_like(Y)
        for i in range(n):
            diff   = Y[i] - Y                       # shape (n, 2)
            factor = (PQ[i] * num[i])[:, None]      # shape (n, 1)
            dY[i]  = 4 * (factor * diff).sum(axis=0)

        # Adaptive learning rate per parameter (sign-based gains)
        gains = (gains + 0.2) * ((dY > 0) != (Y - Y_old > 0)) + \
                (gains * 0.8) * ((dY > 0) == (Y - Y_old > 0))
        gains = np.maximum(gains, 0.01)

        Y_new  = Y - lr * gains * dY + momentum * (Y - Y_old)
        Y_old  = Y.copy()
        Y      = Y_new

        if it == 100:
            P        /= 4.0   # end early exaggeration
            momentum  = 0.8

        if (it + 1) % 200 == 0:
            kl = np.sum(P * np.log(P / Q + 1e-9))
            print(f"  t-SNE iter {it+1:4d} | KL divergence: {kl:.4f}")

    return Y



# similarity helpers

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a)* np.linalg.norm(b) + 1e-9)


def most_similar(word, embeddings, word2idx, idx2word, topn=5):
    if word not in word2idx:
        return[]
    vec = embeddings[word2idx[word]]
    sims = [(idx2word[i], cosine_similarity(vec, embeddings[i])) for i in range(len(idx2word)) if idx2word[i] != word]
    return sorted(sims, key=lambda x: -x[1])[:topn]



# Training both Models

EMBED_DIM = 10
EPOCHS = 300 
LR = 0.05

print("\n" + "="*50)
print("Training Skip-Gram....")
print("="*50)

sg_model = SkipGram(V, EMBED_DIM)
sg_losses = sg_model.train(skipgram_pairs, epochs=EPOCHS, lr=LR)


print("\n" + "="*50)
print("Training CBOW....")
print("="*50)

cbow_model = CBOW(V, EMBED_DIM)
cbow_losses = cbow_model.train(cbow_pairs, epochs=EPOCHS, lr=LR)



# similarity inspection
print("\n" + "="*50)
print("Word similarities (Skip-gram)")
print("="*50)

for word in ["king","queen","france","animals"]:
    if word in word2idx:
        neighbors = most_similar(word, sg_model.embeddings, word2idx, idx2word)
        print(f"\n  '{word}' -> {[(w, round(s,3)) for w,s in neighbors]}")


print("\n" + "="*50)
print("Word similarities (CBOW)")
print("="*50)

for word in ["king","queen","france","animals"]: 
    if word in word2idx:
        neighbors = most_similar(word, cbow_model.embeddings, word2idx, idx2word)
        print(f"\n  '{word}' -> {[(w, round(s,3)) for w,s in neighbors]}")




# Group words by semantic category for color-coding
categories = {
    "royalty"   : ["king", "queen", "prince", "princess"],
    "people"    : ["man", "woman", "president"],
    "animals"   : ["animals", "lion", "dog", "cat", "dogs", "cats",
                   "lions", "tigers"],
    "countries" : ["france", "germany", "italy", "spain", "europe"],
    "cities"    : ["paris", "berlin", "rome"],
    "action"    : ["loves", "leads", "rules", "chased", "become"],
    "other"     : [],
}

# Assign each word a category
word_category = {}
for cat, words in categories.items():
    for w in words:
        if w in word2idx:
            word_category[w] = cat
for w in vocab:
    if w not in word_category:
        word_category[w] = "other"

cat_names   = list(categories.keys())
cat_colors  = cm.tab10(np.linspace(0, 0.9, len(cat_names)))
cat_color_map = {c: cat_colors[i] for i, c in enumerate(cat_names)}


def plot_tsne(embeddings, title, ax, words, word_category, cat_color_map):
    reduced = tsne(embeddings, n_components=2, perplexity=5,
                   lr=150, n_iter=800)

    for i, word in enumerate(words):
        cat   = word_category.get(word, "other")
        color = cat_color_map[cat]
        ax.scatter(reduced[i, 0], reduced[i, 1],
                   color=color, s=80, zorder=3, alpha=0.9)
        ax.annotate(word, (reduced[i, 0], reduced[i, 1]),
                    fontsize=7.5, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points',
                    color='#222')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.set_facecolor('#f8f8f8')


# Legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=cat_color_map[c], label=c) for c in cat_names
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('white')
fig.suptitle("Word2Vec Embeddings — t-SNE Visualization",
             fontsize=16, fontweight='bold', y=1.01)

# ── Panel 1: Skip-gram t-SNE ──
print("\nt-SNE for Skip-gram embeddings...")
plot_tsne(sg_model.embeddings,  "Skip-gram", axes[0],
          vocab, word_category, cat_color_map)

# ── Panel 2: CBOW t-SNE ──
print("\nt-SNE for CBOW embeddings...")
plot_tsne(cbow_model.embeddings, "CBOW", axes[1],
          vocab, word_category, cat_color_map)

# ── Panel 3: Training loss curves ──
ax = axes[2]
ax.plot(sg_losses,   color='#e05c3a', lw=2, label='Skip-gram')
ax.plot(cbow_losses, color='#3a8ee0', lw=2, label='CBOW')
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Avg Cross-Entropy Loss", fontsize=11)
ax.set_title("Training Loss", fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=10)
ax.spines[['top','right']].set_visible(False)
ax.set_facecolor('#f8f8f8')

fig.legend(handles=legend_elements, loc='lower center',
           ncol=len(cat_names), fontsize=9,
           bbox_to_anchor=(0.5, -0.06), frameon=False)

plt.tight_layout()
plt.savefig("word2vec_tsne.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to word2vec_tsne.png")
