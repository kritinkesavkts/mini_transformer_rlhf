#!/usr/bin/env python3
"""
================================================================================
  Mini-Transformer (Decoder-Only) with Complete RLHF Pipeline — V2
  =================================================================
  A from-scratch, single-file implementation in pure PyTorch.

  What makes this version production-methodology:
  ───────────────────────────────────────────────
  • 50+ recipe→summary training pairs (diverse cuisines & complexity)
  • After SFT, we GENERATE candidate summaries from the trained model
  • Those real model outputs get ANNOTATED (scored / compared)
  • The Reward Model trains on comparisons of REAL generations
  • PPO-style RL uses the trained RM as an automated judge
  • Full evaluation: side-by-side tables + training plots saved as PNGs

  Pipeline Flow
  ─────────────
  Phase 1 │ Supervised Fine-Tuning (SFT)
          │   Train on 50+ recipe→summary pairs with teacher forcing
          ▼
  Phase 2 │ Candidate Generation
          │   Generate multiple summaries per prompt from the SFT model
          ▼
  Phase 3 │ Annotation / Preference Labelling
          │   Score & rank the generated candidates (heuristic-simulated)
          │   Create pairwise comparisons: (prompt, preferred, rejected)
          ▼
  Phase 4 │ Reward Model Training
          │   Train RM on real comparison pairs from Phase 3
          ▼
  Phase 5 │ PPO-Inspired RL Fine-Tuning
          │   Policy generates → RM scores → KL penalty → update policy
          ▼
  Phase 6 │ Evaluation & Visualisation
          │   Side-by-side table, reward distributions, training curves

  Run:  python mini_transformer_rlhf_v2.py
  Deps: torch, matplotlib (pip install torch matplotlib)
================================================================================
"""

import math
import copy
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import defaultdict

# Matplotlib — optional for environments without display
try:
    import matplotlib
    matplotlib.use("Agg")        # non-interactive backend (safe for Colab & servers)
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠  matplotlib not found — skipping plot generation.")


# ══════════════════════════════════════════════════════════════════════════════
# 0.  HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
D_MODEL       = 128        # transformer hidden dimension
N_HEADS       = 4          # attention heads
N_LAYERS      = 3          # transformer blocks (up from 2 for larger dataset)
D_FF          = 256        # feed-forward inner dim
DROPOUT       = 0.1
MAX_SEQ_LEN   = 160        # max sequence length
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Phase 1: SFT
SFT_EPOCHS    = 400
SFT_LR        = 3e-4

# Phase 4: Reward Model
RM_EPOCHS     = 300
RM_LR         = 1e-4

# Phase 5: RL (PPO-inspired)
RL_STEPS      = 50
RL_LR         = 1e-5
KL_BETA       = 0.15       # KL divergence penalty weight
GEN_MAX_LEN   = 50         # max tokens per RL generation
TEMPERATURE   = 0.8        # sampling temperature during RL rollouts

# Phase 2: Generation
NUM_CANDIDATES = 4          # summaries to generate per prompt
GEN_TEMPERATURES = [0.5, 0.7, 0.9, 1.1]  # different temps for diversity

# Reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Output directory for plots
OUTPUT_DIR = "rlhf_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET — 50+ Recipe → Summary Pairs
# ══════════════════════════════════════════════════════════════════════════════
# Each pair: short recipe instruction → concise summary.
# Diverse cuisines, techniques, and complexity levels.

RECIPES = [
    # ── Baking ────────────────────────────────────────────────────────────────
    {"recipe": "Mix flour, sugar, eggs, butter. Bake at 350F for 25 min.",
     "summary": "Simple cake: mix basics, bake 25 min."},
    {"recipe": "Combine flour, yeast, water, salt. Knead 10 min. Rise 1 hour. Bake 400F 30 min.",
     "summary": "Basic bread: knead, rise 1 hr, bake 30 min."},
    {"recipe": "Cream butter and sugar. Add eggs, vanilla. Fold in flour. Bake 375F 12 min.",
     "summary": "Vanilla cookies: cream, fold, bake 12 min."},
    {"recipe": "Melt chocolate, mix with cream. Chill 2 hours. Roll into balls.",
     "summary": "Chocolate truffles: melt, chill, roll."},
    {"recipe": "Whisk eggs, sugar, lemon juice. Cook over double boiler until thick.",
     "summary": "Lemon curd: whisk and cook until thick."},
    {"recipe": "Mix oats, honey, peanut butter. Press into pan. Refrigerate 1 hour.",
     "summary": "No-bake oat bars: mix, press, chill."},
    {"recipe": "Combine graham crumbs, butter for crust. Fill with cream cheese mix. Bake 325F.",
     "summary": "Cheesecake: graham crust, cream cheese, bake."},
    {"recipe": "Sift flour, cocoa, sugar. Add eggs, oil, coffee. Bake 350F 35 min.",
     "summary": "Chocolate cake with coffee, bake 35 min."},

    # ── Pasta & Italian ───────────────────────────────────────────────────────
    {"recipe": "Boil pasta. Fry garlic in oil. Toss together with parmesan.",
     "summary": "Quick garlic parmesan pasta."},
    {"recipe": "Cook spaghetti. Fry pancetta. Mix eggs and pecorino. Combine off heat.",
     "summary": "Carbonara: pancetta, egg, pecorino."},
    {"recipe": "Simmer tomatoes, garlic, basil for 20 min. Toss with penne.",
     "summary": "Penne in tomato basil sauce."},
    {"recipe": "Layer lasagna noodles, ricotta, meat sauce, mozzarella. Bake 375F 45 min.",
     "summary": "Classic lasagna: layer and bake 45 min."},
    {"recipe": "Blend basil, pine nuts, garlic, parmesan, olive oil. Toss with pasta.",
     "summary": "Fresh basil pesto pasta."},
    {"recipe": "Saute shrimp with garlic butter. Toss with linguine and lemon.",
     "summary": "Garlic butter shrimp linguine."},

    # ── Asian ─────────────────────────────────────────────────────────────────
    {"recipe": "Dice chicken. Stir fry with soy sauce, ginger, and veggies.",
     "summary": "Chicken stir fry with soy and ginger."},
    {"recipe": "Marinate tofu in soy and sesame. Pan fry until crispy. Serve with rice.",
     "summary": "Crispy sesame tofu with rice."},
    {"recipe": "Cook rice. Fry egg, add rice, soy sauce, peas, green onion.",
     "summary": "Egg fried rice with peas and onion."},
    {"recipe": "Simmer chicken broth with ginger, noodles, bok choy, and chili oil.",
     "summary": "Ginger chicken noodle soup with bok choy."},
    {"recipe": "Mix ground pork, cabbage, ginger. Wrap in dumpling skins. Steam 10 min.",
     "summary": "Pork cabbage dumplings, steamed 10 min."},
    {"recipe": "Stir fry beef with broccoli in oyster sauce. Serve over rice.",
     "summary": "Beef broccoli in oyster sauce."},
    {"recipe": "Roll sushi rice and fish in nori. Slice into pieces.",
     "summary": "Basic sushi rolls with fish."},
    {"recipe": "Fry rice noodles with shrimp, egg, bean sprouts, and tamarind sauce.",
     "summary": "Shrimp pad thai with tamarind."},
    {"recipe": "Simmer coconut milk, curry paste, chicken, bamboo shoots. Serve with rice.",
     "summary": "Thai coconut curry chicken."},

    # ── Mexican & Latin ───────────────────────────────────────────────────────
    {"recipe": "Season beef with cumin and chili. Cook with onions. Serve in tortillas.",
     "summary": "Beef tacos with cumin and chili."},
    {"recipe": "Mash avocado with lime, salt, cilantro, onion, jalapeno.",
     "summary": "Fresh guacamole with lime and cilantro."},
    {"recipe": "Layer tortillas, chicken, salsa, cheese. Bake 375F 20 min.",
     "summary": "Chicken enchilada bake, 20 min."},
    {"recipe": "Cook black beans with cumin, garlic, lime. Serve over rice.",
     "summary": "Cuban black beans and rice."},
    {"recipe": "Grill corn, add mayo, chili powder, lime, cotija cheese.",
     "summary": "Mexican street corn with cotija."},

    # ── Breakfast ─────────────────────────────────────────────────────────────
    {"recipe": "Blend banana, milk, honey, ice for a smoothie.",
     "summary": "Banana honey smoothie."},
    {"recipe": "Spread peanut butter on toast. Add banana slices and honey.",
     "summary": "PB banana honey toast."},
    {"recipe": "Whisk eggs, pour in hot pan. Add cheese, fold in half.",
     "summary": "Cheese omelette: whisk, pour, fold."},
    {"recipe": "Soak oats in yogurt overnight. Top with berries and honey.",
     "summary": "Overnight oats with berries."},
    {"recipe": "Mix flour, eggs, milk, sugar. Pour in hot pan. Flip when bubbly.",
     "summary": "Fluffy pancakes: mix, pour, flip."},
    {"recipe": "Scramble eggs with spinach, feta, and sun dried tomatoes.",
     "summary": "Mediterranean scrambled eggs."},
    {"recipe": "Blend spinach, banana, protein powder, almond milk.",
     "summary": "Green protein smoothie."},
    {"recipe": "Toast bread, layer avocado, egg, red pepper flakes, salt.",
     "summary": "Avocado egg toast with pepper flakes."},

    # ── Soups & Stews ─────────────────────────────────────────────────────────
    {"recipe": "Simmer chicken, carrots, celery, noodles in broth for 30 min.",
     "summary": "Classic chicken noodle soup, 30 min."},
    {"recipe": "Roast butternut squash. Blend with broth, cream, nutmeg.",
     "summary": "Roasted butternut squash soup."},
    {"recipe": "Brown beef, add potatoes, carrots, broth. Simmer 2 hours.",
     "summary": "Beef stew: brown, simmer 2 hours."},
    {"recipe": "Saute onion, add lentils, tomatoes, cumin, broth. Cook 25 min.",
     "summary": "Cumin lentil tomato soup."},
    {"recipe": "Cook bacon, add corn, potatoes, cream, thyme. Simmer 20 min.",
     "summary": "Bacon corn chowder with thyme."},

    # ── Salads & Light ────────────────────────────────────────────────────────
    {"recipe": "Toss romaine, croutons, parmesan with caesar dressing.",
     "summary": "Classic caesar salad."},
    {"recipe": "Combine quinoa, cucumber, tomato, feta, lemon vinaigrette.",
     "summary": "Quinoa feta salad with lemon."},
    {"recipe": "Mix chickpeas, cucumber, red onion, parsley, lemon, olive oil.",
     "summary": "Mediterranean chickpea salad."},
    {"recipe": "Layer sliced tomato, mozzarella, basil. Drizzle balsamic and oil.",
     "summary": "Caprese salad with balsamic."},

    # ── Grilling & Meat ───────────────────────────────────────────────────────
    {"recipe": "Season steak with salt, pepper. Grill 4 min each side. Rest 5 min.",
     "summary": "Grilled steak: season, grill 4 min per side."},
    {"recipe": "Marinate chicken in yogurt and spices. Grill until charred.",
     "summary": "Yogurt-marinated grilled chicken."},
    {"recipe": "Rub ribs with brown sugar, paprika, garlic. Slow cook 6 hours.",
     "summary": "Slow cooked ribs with spice rub."},
    {"recipe": "Mix ground beef, breadcrumbs, egg, onion. Shape and grill.",
     "summary": "Classic beef burgers, grilled."},
    {"recipe": "Butterfly chicken breast. Stuff with spinach and cheese. Bake 375F.",
     "summary": "Spinach cheese stuffed chicken."},

    # ── Seafood ───────────────────────────────────────────────────────────────
    {"recipe": "Season salmon with lemon and dill. Bake 400F for 15 min.",
     "summary": "Lemon dill baked salmon, 15 min."},
    {"recipe": "Saute shrimp in garlic, white wine, butter. Serve with bread.",
     "summary": "Shrimp scampi in garlic wine butter."},
    {"recipe": "Coat fish in flour, fry in oil. Serve with tartar sauce.",
     "summary": "Crispy fried fish with tartar sauce."},

    # ── Snacks & Sides ────────────────────────────────────────────────────────
    {"recipe": "Toss chickpeas with oil and spices. Roast 400F 25 min until crispy.",
     "summary": "Crispy roasted spiced chickpeas."},
    {"recipe": "Slice potatoes thin. Layer with cream and cheese. Bake 375F 1 hour.",
     "summary": "Potato gratin: layer, bake 1 hour."},
    {"recipe": "Mash sweet potatoes with butter, maple syrup, cinnamon.",
     "summary": "Maple cinnamon mashed sweet potatoes."},
    {"recipe": "Toss broccoli with oil, garlic. Roast 425F 20 min.",
     "summary": "Roasted garlic broccoli."},
    {"recipe": "Mix hummus: blend chickpeas, tahini, lemon, garlic, olive oil.",
     "summary": "Classic hummus with tahini and lemon."},
]

# We'll hold out the last 10 recipes for evaluation
TRAIN_RECIPES = RECIPES[:-10]
EVAL_RECIPES  = RECIPES[-10:]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CHARACTER-LEVEL TOKENISER
# ══════════════════════════════════════════════════════════════════════════════
class CharTokeniser:
    """
    Character-level tokeniser with special tokens.
    Keeps the demo fully self-contained — no external tokeniser needed.

    Special tokens:
        <PAD> = 0    padding
        <BOS> = 1    beginning of sequence
        <EOS> = 2    end of sequence
        <SEP> = 3    separates recipe from summary
    """
    def __init__(self):
        self.special = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<SEP>": 3}
        self.char2idx = {chr(i): i - 32 + 4 for i in range(32, 127)}
        self.char2idx.update(self.special)
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.vocab_size = max(self.idx2char.keys()) + 1

    def encode(self, text: str) -> list[int]:
        return [self.char2idx.get(c, self.special["<PAD>"]) for c in text]

    def decode(self, ids: list[int]) -> str:
        out = []
        for idx in ids:
            tok = self.idx2char.get(idx, "")
            if tok in ("<PAD>", "<BOS>", "<EOS>", "<SEP>"):
                continue
            out.append(tok)
        return "".join(out)

    def build_sample(self, recipe: str, summary: str) -> list[int]:
        """Encode: <BOS> recipe <SEP> summary <EOS>"""
        return (
            [self.special["<BOS>"]]
            + self.encode(recipe)
            + [self.special["<SEP>"]]
            + self.encode(summary)
            + [self.special["<EOS>"]]
        )

    def build_prompt(self, recipe: str) -> list[int]:
        """Encode just the prompt part: <BOS> recipe <SEP>"""
        return (
            [self.special["<BOS>"]]
            + self.encode(recipe)
            + [self.special["<SEP>"]]
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRANSFORMER — Built from Scratch
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention with causal masking.
    Each head independently learns different attention patterns.
    The causal mask ensures position i can only attend to positions ≤ i.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(context)


class FeedForward(nn.Module):
    """Position-wise FFN: Linear → GELU → Dropout → Linear → Dropout"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x): return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm decoder block: LN→MHA→+res → LN→FFN→+res"""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    """
    GPT-style decoder-only transformer.

    Architecture:
        Token Embedding + Learned Positional Embedding
        → N × TransformerBlock (Pre-Norm)
        → Final LayerNorm
        → Linear Head (weight-tied with token embedding)
    """
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ff=D_FF, max_len=MAX_SEQ_LEN,
                 dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def get_hidden(self, idx):
        """Return the final hidden states (before the LM head)."""
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=50, eos_id=2, temperature=0.7):
        """Autoregressive generation with temperature sampling."""
        self.eval()
        gen = prompt_ids.clone()
        for _ in range(max_new):
            logits = self.forward(gen[:, -MAX_SEQ_LEN:])
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs  = F.softmax(logits, dim=-1)
            nxt    = torch.multinomial(probs, 1)
            gen    = torch.cat([gen, nxt], dim=1)
            if nxt.item() == eos_id:
                break
        return gen

    @torch.no_grad()
    def generate_greedy(self, prompt_ids, max_new=50, eos_id=2):
        """Greedy (argmax) generation — deterministic."""
        self.eval()
        gen = prompt_ids.clone()
        for _ in range(max_new):
            logits = self.forward(gen[:, -MAX_SEQ_LEN:])
            nxt    = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen    = torch.cat([gen, nxt], dim=1)
            if nxt.item() == eos_id:
                break
        return gen


# ══════════════════════════════════════════════════════════════════════════════
# 4.  REWARD MODEL
# ══════════════════════════════════════════════════════════════════════════════
class RewardModel(nn.Module):
    """
    Same backbone as MiniTransformer but with a scalar head instead of
    the LM head.  Outputs a single reward value r ∈ ℝ for a full sequence.

    Training uses the Bradley-Terry comparison loss:
        L = -log σ(r_preferred - r_rejected)
    """
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ff=D_FF, max_len=MAX_SEQ_LEN,
                 dropout=DROPOUT):
        super().__init__()
        self.backbone    = MiniTransformer(vocab_size, d_model, n_heads,
                                           n_layers, d_ff, max_len, dropout)
        self.scalar_head = nn.Linear(d_model, 1)

    def forward(self, idx):
        hidden = self.backbone.get_hidden(idx)   # (B, T, d_model)
        last_h = hidden[:, -1, :]                # (B, d_model)
        # ★ SCALAR HEAD — this single number drives the RL policy gradient
        return self.scalar_head(last_h)           # (B, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def pad_and_batch(sequences, pad_id=0):
    """Pad variable-length sequences and return (B, T) tensor."""
    max_len = min(max(len(s) for s in sequences), MAX_SEQ_LEN)
    padded  = [s[:max_len] + [pad_id] * max(0, max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long, device=DEVICE)


def sequence_log_probs(model, token_ids):
    """
    Per-token log-probabilities: log π(a_t | s_{<t}).
    Returns shape (batch, seq_len - 1).
    """
    logits  = model(token_ids)                         # (B, T, V)
    log_p   = F.log_softmax(logits[:, :-1, :], dim=-1) # (B, T-1, V)
    targets = token_ids[:, 1:].unsqueeze(-1)            # (B, T-1, 1)
    return log_p.gather(-1, targets).squeeze(-1)        # (B, T-1)


def kl_div_per_token(log_probs_policy, log_probs_ref):
    """
    Approximate per-token KL: D_KL(π_policy ‖ π_ref) ≈ log π_policy - log π_ref
    Prevents reward hacking by penalising policy drift from SFT reference.
    """
    return log_probs_policy - log_probs_ref


def print_header(title):
    """Pretty section header."""
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PHASE 1 — SUPERVISED FINE-TUNING (SFT)
# ══════════════════════════════════════════════════════════════════════════════

def train_sft(model, tokeniser):
    """
    Phase 1: Standard causal LM training with cross-entropy loss.

    The model learns to predict each next character in:
        <BOS> recipe_text <SEP> summary_text <EOS>
    """
    print_header("PHASE 1: Supervised Fine-Tuning (SFT)")
    print(f"  Training on {len(TRAIN_RECIPES)} recipe-summary pairs")
    print(f"  Epochs: {SFT_EPOCHS}  |  LR: {SFT_LR}")

    model.train()
    optimiser = Adam(model.parameters(), lr=SFT_LR)

    # Prepare all training samples
    samples = [tokeniser.build_sample(r["recipe"], r["summary"])
               for r in TRAIN_RECIPES]

    # Mini-batch training (batch size = 8)
    batch_size = 8
    loss_history = []

    for epoch in range(1, SFT_EPOCHS + 1):
        random.shuffle(samples)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, len(samples), batch_size):
            batch = pad_and_batch(samples[i:i + batch_size])
            inputs  = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(inputs)
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=0,
            )

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  [SFT]  epoch {epoch:>4d}/{SFT_EPOCHS}  loss = {avg_loss:.4f}")

    # Quick generation check
    print("\n  ── SFT Generation Check (greedy) ──")
    model.eval()
    for r in TRAIN_RECIPES[:3]:
        prompt_t = torch.tensor([tokeniser.build_prompt(r["recipe"])], device=DEVICE)
        gen = model.generate_greedy(prompt_t, max_new=60)
        print(f"  Recipe  : {r['recipe'][:60]}...")
        print(f"  Target  : {r['summary']}")
        print(f"  SFT gen : {tokeniser.decode(gen[0].tolist())}")
        print()

    return loss_history


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PHASE 2 — CANDIDATE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_candidates(model, tokeniser):
    """
    Phase 2: Generate multiple summary candidates for each recipe prompt.

    We sample at different temperatures to get diverse outputs:
        - Low temp (0.5)  → conservative, close to training data
        - High temp (1.1) → creative, possibly garbled

    These become the "real model outputs" that get annotated in Phase 3.
    In production, you would present these to human annotators.
    """
    print_header("PHASE 2: Generating Candidate Summaries")
    print(f"  Generating {NUM_CANDIDATES} candidates per prompt "
          f"at temps {GEN_TEMPERATURES}")

    model.eval()
    all_candidates = []  # list of dicts: {recipe, candidates: [...]}

    for r in TRAIN_RECIPES:
        prompt_t = torch.tensor([tokeniser.build_prompt(r["recipe"])], device=DEVICE)
        candidates = []

        for temp in GEN_TEMPERATURES:
            gen = model.generate(prompt_t, max_new=GEN_MAX_LEN, temperature=temp)
            text = tokeniser.decode(gen[0].tolist())
            candidates.append({
                "text": text,
                "temperature": temp,
            })

        all_candidates.append({
            "recipe": r["recipe"],
            "reference": r["summary"],
            "candidates": candidates,
        })

    # Show some examples
    print(f"\n  Generated candidates for {len(all_candidates)} recipes.")
    print("\n  ── Sample Candidates ──")
    for item in all_candidates[:3]:
        print(f"\n  Recipe: {item['recipe'][:65]}")
        print(f"  Reference: {item['reference']}")
        for i, c in enumerate(item["candidates"]):
            print(f"    Candidate {i+1} (T={c['temperature']}): {c['text'][:70]}")

    return all_candidates


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PHASE 3 — ANNOTATION (Simulated Human Preferences)
# ══════════════════════════════════════════════════════════════════════════════

def compute_quality_score(candidate_text, reference_text, recipe_text):
    """
    Heuristic quality score simulating a human annotator.

    In a real RLHF pipeline, this function would be replaced by actual
    human ratings (e.g., from Scale AI, Surge AI, or an internal team).
    
    We use multiple signals that a human annotator would also consider:
      1. Keyword overlap with reference  (content accuracy)
      2. Length appropriateness           (conciseness)
      3. Key ingredient/method coverage   (completeness)
      4. Readability                      (not garbled)

    Returns a float score (higher = better).
    """
    score = 0.0
    cand_lower = candidate_text.lower()
    ref_lower  = reference_text.lower()
    rec_lower  = recipe_text.lower()

    # 1. Keyword overlap with reference summary
    ref_words  = set(ref_lower.split())
    cand_words = set(cand_lower.split())
    if len(ref_words) > 0:
        overlap = len(ref_words & cand_words) / len(ref_words)
        score += overlap * 3.0     # up to 3 points

    # 2. Length appropriateness (summaries should be concise: 20–60 chars)
    cand_len = len(candidate_text)
    if 15 <= cand_len <= 70:
        score += 2.0               # good length
    elif 10 <= cand_len <= 100:
        score += 1.0               # acceptable
    else:
        score -= 1.0               # too short or too long

    # 3. Key ingredient/method coverage from the recipe
    key_words = []
    for w in rec_lower.split():
        if len(w) > 3 and w not in {"with", "into", "from", "over", "until", "about"}:
            key_words.append(w)
    if key_words:
        coverage = sum(1 for w in key_words if w in cand_lower) / len(key_words)
        score += coverage * 2.0    # up to 2 points

    # 4. Readability / non-garbled check
    # Penalise repeated characters or very short outputs
    if len(candidate_text.strip()) < 5:
        score -= 3.0               # empty or garbled
    if len(set(candidate_text)) < 5:
        score -= 2.0               # very low character diversity → likely garbled

    return score


def annotate_candidates(all_candidates):
    """
    Phase 3: Score and rank generated candidates, then build comparison pairs.

    This simulates the human annotation step:
      1. Score each candidate using quality heuristics
      2. For each recipe, pick the best and worst candidates
      3. Create (prompt, preferred, rejected) comparison pairs

    Output: List of dicts with keys: recipe, preferred, rejected, scores
    """
    print_header("PHASE 3: Annotating Generated Candidates")

    comparison_pairs = []
    annotation_log   = []

    for item in all_candidates:
        recipe    = item["recipe"]
        reference = item["reference"]
        scored    = []

        for c in item["candidates"]:
            quality = compute_quality_score(c["text"], reference, recipe)
            scored.append({
                "text":  c["text"],
                "temp":  c["temperature"],
                "score": quality,
            })

        # Sort by score (highest first)
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Create comparison pair: best vs worst
        if len(scored) >= 2 and scored[0]["score"] != scored[-1]["score"]:
            pair = {
                "recipe":    recipe,
                "preferred": scored[0]["text"],
                "rejected":  scored[-1]["text"],
                "pref_score": scored[0]["score"],
                "rej_score":  scored[-1]["score"],
            }
            comparison_pairs.append(pair)

        annotation_log.append({
            "recipe": recipe,
            "scores": [(s["text"][:50], round(s["score"], 2)) for s in scored],
        })

    # Show annotation results
    print(f"  Created {len(comparison_pairs)} comparison pairs from real generations.\n")
    print("  ── Sample Annotations ──")
    for entry in annotation_log[:4]:
        print(f"\n  Recipe: {entry['recipe'][:60]}")
        for text, sc in entry["scores"]:
            marker = "  ★" if sc == max(s for _, s in entry["scores"]) else "   "
            print(f"  {marker} [{sc:+.2f}] {text}...")

    print(f"\n  ── Sample Comparison Pairs ──")
    for pair in comparison_pairs[:3]:
        print(f"\n  Recipe   : {pair['recipe'][:55]}")
        print(f"  Preferred: ({pair['pref_score']:+.2f}) {pair['preferred'][:55]}")
        print(f"  Rejected : ({pair['rej_score']:+.2f}) {pair['rejected'][:55]}")

    return comparison_pairs, annotation_log


# ══════════════════════════════════════════════════════════════════════════════
# 9.  PHASE 4 — REWARD MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_reward_model(reward_model, tokeniser, comparison_pairs):
    """
    Phase 4: Train the Reward Model on REAL comparison pairs.

    Loss: -log σ(r_preferred - r_rejected)   [Bradley-Terry]

    The RM learns to assign higher rewards to summaries that the
    annotation phase marked as preferred.
    """
    print_header("PHASE 4: Reward Model Training on Real Comparisons")
    print(f"  Training on {len(comparison_pairs)} comparison pairs")
    print(f"  Epochs: {RM_EPOCHS}  |  LR: {RM_LR}")

    reward_model.train()
    optimiser = Adam(reward_model.parameters(), lr=RM_LR)
    loss_history = []

    for epoch in range(1, RM_EPOCHS + 1):
        random.shuffle(comparison_pairs)
        epoch_loss = 0.0

        for pair in comparison_pairs:
            pref_ids = tokeniser.build_sample(pair["recipe"], pair["preferred"])
            rej_ids  = tokeniser.build_sample(pair["recipe"], pair["rejected"])

            pref_t = pad_and_batch([pref_ids])
            rej_t  = pad_and_batch([rej_ids])

            r_pref = reward_model(pref_t)   # (1, 1)
            r_rej  = reward_model(rej_t)    # (1, 1)

            # ★ COMPARISON LOSS: push r_preferred above r_rejected
            loss = -F.logsigmoid(r_pref - r_rej).mean()

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
            optimiser.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(comparison_pairs), 1)
        loss_history.append(avg_loss)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  [RM]   epoch {epoch:>4d}/{RM_EPOCHS}  loss = {avg_loss:.4f}")

    # Verify: check ordering accuracy on training pairs
    reward_model.eval()
    correct = 0
    with torch.no_grad():
        for pair in comparison_pairs:
            p_t = pad_and_batch([tokeniser.build_sample(pair["recipe"], pair["preferred"])])
            r_t = pad_and_batch([tokeniser.build_sample(pair["recipe"], pair["rejected"])])
            if reward_model(p_t).item() > reward_model(r_t).item():
                correct += 1

    acc = correct / max(len(comparison_pairs), 1) * 100
    print(f"\n  RM ordering accuracy: {correct}/{len(comparison_pairs)} = {acc:.1f}%")

    # Show some scores
    print("\n  ── RM Score Samples ──")
    with torch.no_grad():
        for pair in comparison_pairs[:3]:
            p_t = pad_and_batch([tokeniser.build_sample(pair["recipe"], pair["preferred"])])
            r_t = pad_and_batch([tokeniser.build_sample(pair["recipe"], pair["rejected"])])
            rp  = reward_model(p_t).item()
            rr  = reward_model(r_t).item()
            print(f"  Preferred [{rp:+.3f}]: {pair['preferred'][:50]}")
            print(f"  Rejected  [{rr:+.3f}]: {pair['rejected'][:50]}")
            print(f"  {'✓ Correct' if rp > rr else '✗ Wrong'}  Δ = {rp - rr:.3f}\n")

    return loss_history


# ══════════════════════════════════════════════════════════════════════════════
# 10.  PHASE 5 — PPO-INSPIRED RL FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════

def rl_training_loop(policy_model, ref_model, reward_model, tokeniser):
    """
    Phase 5: PPO-inspired RL loop.

    For each step:
      1. Sample a recipe prompt
      2. Generate a summary from the RL policy (temperature sampling)
      3. Score it with the frozen Reward Model  → reward scalar r
      4. Compute log-probs under policy AND frozen SFT reference
      5. KL penalty = log π_policy - log π_ref  (per token)
      6. Advantage = reward - β × mean(KL)
      7. Loss = -(log_probs × advantage) + β × KL
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         ★ The reward scalar enters the gradient via `advantage`:
              ∇θ loss  ∝  -advantage × ∇θ log π_θ
           High reward → large positive advantage → REINFORCE tokens
           Low reward  → negative advantage → SUPPRESS tokens
         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    print_header("PHASE 5: PPO-Inspired RL Fine-Tuning")
    print(f"  RL steps: {RL_STEPS}  |  LR: {RL_LR}  |  KL β: {KL_BETA}")

    policy_model.train()
    ref_model.eval()
    reward_model.eval()

    optimiser = Adam(policy_model.parameters(), lr=RL_LR)

    # Tracking metrics
    reward_history = []
    kl_history     = []
    loss_history   = []

    for step in range(1, RL_STEPS + 1):
        # 1. Random prompt ─────────────────────────────────────────────────────
        recipe = TRAIN_RECIPES[step % len(TRAIN_RECIPES)]
        prompt_ids = tokeniser.build_prompt(recipe["recipe"])
        prompt_t   = torch.tensor([prompt_ids], device=DEVICE)

        # 2. Generate from policy (sampling, with exploration) ─────────────────
        policy_model.eval()
        with torch.no_grad():
            generated = prompt_t.clone()
            for _ in range(GEN_MAX_LEN):
                logits = policy_model(generated[:, -MAX_SEQ_LEN:])
                probs  = F.softmax(logits[:, -1, :] / TEMPERATURE, dim=-1)
                nxt    = torch.multinomial(probs, 1)
                generated = torch.cat([generated, nxt], dim=1)
                if nxt.item() == tokeniser.special["<EOS>"]:
                    break
        policy_model.train()

        # 3. Score with frozen Reward Model ────────────────────────────────────
        with torch.no_grad():
            reward_scalar = reward_model(generated).item()
            # ↑ This is the scalar r that drives the policy gradient

        # 4. Log-probs under both models ───────────────────────────────────────
        lp_policy = sequence_log_probs(policy_model, generated)   # (1, T-1)
        with torch.no_grad():
            lp_ref = sequence_log_probs(ref_model, generated)     # (1, T-1)

        # 5. KL penalty ────────────────────────────────────────────────────────
        kl = kl_div_per_token(lp_policy, lp_ref)
        mean_kl = kl.mean()

        # 6. Advantage  (reward − β × KL) ─────────────────────────────────────
        #    ★ reward_scalar is a Python float — it has NO gradient of its own.
        #      It scales the log-prob gradient as a REINFORCE coefficient:
        #         ∂loss/∂θ = -advantage × ∂(log π_θ)/∂θ
        advantage = reward_scalar - KL_BETA * mean_kl.detach().item()

        # 7. Policy gradient loss ──────────────────────────────────────────────
        #   ┌──────────────────────────────────────────────────────────────────┐
        #   │ ★ THIS IS WHERE THE REWARD SCALAR INFLUENCES THE GRADIENT.     │
        #   │                                                                  │
        #   │   pg_loss = -(mean_log_prob × advantage)                        │
        #   │                                                                  │
        #   │   During .backward(), PyTorch computes:                         │
        #   │     ∂loss/∂θ = -advantage × ∂(mean_log_prob)/∂θ  + β×∂KL/∂θ   │
        #   │                                                                  │
        #   │   • advantage > 0 (good reward, low KL):                        │
        #   │       gradient REINFORCES the generated tokens                  │
        #   │   • advantage < 0 (bad reward or high KL):                      │
        #   │       gradient SUPPRESSES the generated tokens                  │
        #   └──────────────────────────────────────────────────────────────────┘
        pg_loss = -(lp_policy.mean() * advantage)
        kl_loss = KL_BETA * mean_kl
        loss    = pg_loss + kl_loss

        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimiser.step()

        # Track metrics
        reward_history.append(reward_scalar)
        kl_history.append(mean_kl.item())
        loss_history.append(loss.item())

        gen_text = tokeniser.decode(generated[0].tolist())
        if step % 5 == 0 or step == 1:
            print(f"  [RL]  step {step:>3d}/{RL_STEPS}  "
                  f"reward={reward_scalar:+.3f}  "
                  f"KL={mean_kl.item():.4f}  "
                  f"adv={advantage:+.3f}  "
                  f"loss={loss.item():.4f}")
            print(f"        gen: \"{gen_text[:75]}\"")

    return reward_history, kl_history, loss_history


# ══════════════════════════════════════════════════════════════════════════════
# 11.  PHASE 6 — EVALUATION & VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(sft_model, rl_model, reward_model, tokeniser):
    """
    Phase 6: Side-by-side comparison of SFT vs RL-tuned model.

    For each held-out recipe, generate with both models and score
    with the Reward Model.  Print a comparison table.
    """
    print_header("PHASE 6: Evaluation — SFT vs RL-Tuned Model")

    sft_model.eval()
    rl_model.eval()
    reward_model.eval()

    results = []

    with torch.no_grad():
        for r in EVAL_RECIPES:
            prompt_t = torch.tensor([tokeniser.build_prompt(r["recipe"])], device=DEVICE)

            # Generate from both models (greedy for fair comparison)
            sft_gen = sft_model.generate_greedy(prompt_t, max_new=60)
            rl_gen  = rl_model.generate_greedy(prompt_t, max_new=60)

            sft_text = tokeniser.decode(sft_gen[0].tolist())
            rl_text  = tokeniser.decode(rl_gen[0].tolist())

            # Score both with the RM
            sft_reward = reward_model(sft_gen).item()
            rl_reward  = reward_model(rl_gen).item()

            results.append({
                "recipe":     r["recipe"],
                "reference":  r["summary"],
                "sft_text":   sft_text,
                "rl_text":    rl_text,
                "sft_reward": sft_reward,
                "rl_reward":  rl_reward,
            })

    # Print comparison table
    print(f"\n  {'─' * 100}")
    print(f"  {'Recipe':<35} │ {'SFT Output':<30} │ {'RL Output':<30} │ SFT R │ RL R  │ Δ")
    print(f"  {'─' * 100}")
    
    sft_wins = 0
    rl_wins  = 0

    for res in results:
        delta = res["rl_reward"] - res["sft_reward"]
        marker = "▲" if delta > 0 else "▼" if delta < 0 else "="
        if delta > 0:
            rl_wins += 1
        elif delta < 0:
            sft_wins += 1

        print(f"  {res['recipe'][:35]:<35} │ "
              f"{res['sft_text'][:30]:<30} │ "
              f"{res['rl_text'][:30]:<30} │ "
              f"{res['sft_reward']:+.2f} │ "
              f"{res['rl_reward']:+.2f} │ {marker}{abs(delta):.2f}")

    print(f"  {'─' * 100}")
    avg_sft = sum(r["sft_reward"] for r in results) / len(results)
    avg_rl  = sum(r["rl_reward"] for r in results) / len(results)
    print(f"\n  Average SFT reward: {avg_sft:+.3f}")
    print(f"  Average RL reward:  {avg_rl:+.3f}")
    print(f"  RL wins: {rl_wins}/{len(results)}  |  SFT wins: {sft_wins}/{len(results)}")

    return results


def plot_training_curves(sft_loss, rm_loss, rl_rewards, rl_kl, rl_loss):
    """Save training plots to PNG files."""
    if not HAS_MATPLOTLIB:
        print("  ⚠  Skipping plots (matplotlib not available)")
        return

    print(f"\n  Saving plots to {OUTPUT_DIR}/...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("RLHF Training Pipeline — All Phases", fontsize=16, fontweight="bold")

    # 1. SFT Loss
    axes[0, 0].plot(sft_loss, color="#2196F3", linewidth=1.5)
    axes[0, 0].set_title("Phase 1: SFT Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-Entropy Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. RM Loss
    axes[0, 1].plot(rm_loss, color="#FF9800", linewidth=1.5)
    axes[0, 1].set_title("Phase 4: Reward Model Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Comparison Loss")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. RL Rewards over steps
    axes[0, 2].plot(rl_rewards, color="#4CAF50", linewidth=2, marker="o", markersize=3)
    axes[0, 2].set_title("Phase 5: Reward per RL Step")
    axes[0, 2].set_xlabel("RL Step")
    axes[0, 2].set_ylabel("Reward (from RM)")
    axes[0, 2].grid(True, alpha=0.3)
    # Add trend line
    if len(rl_rewards) > 2:
        z = torch.tensor(rl_rewards, dtype=torch.float)
        window = min(5, len(rl_rewards))
        smoothed = [z[max(0, i - window):i + 1].mean().item()
                     for i in range(len(z))]
        axes[0, 2].plot(smoothed, color="#4CAF50", alpha=0.4, linewidth=4)

    # 4. KL Divergence
    axes[1, 0].plot(rl_kl, color="#F44336", linewidth=1.5)
    axes[1, 0].set_title("Phase 5: KL Divergence (Policy vs SFT)")
    axes[1, 0].set_xlabel("RL Step")
    axes[1, 0].set_ylabel("Mean KL")
    axes[1, 0].grid(True, alpha=0.3)

    # 5. RL Loss
    axes[1, 1].plot(rl_loss, color="#9C27B0", linewidth=1.5)
    axes[1, 1].set_title("Phase 5: RL Total Loss")
    axes[1, 1].set_xlabel("RL Step")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Pipeline Summary (text box)
    axes[1, 2].axis("off")
    summary_text = (
        "Pipeline Summary\n"
        "─────────────────\n"
        f"SFT: {len(TRAIN_RECIPES)} recipes, {SFT_EPOCHS} epochs\n"
        f"Final SFT loss: {sft_loss[-1]:.4f}\n\n"
        f"RM: trained on real generations\n"
        f"Final RM loss: {rm_loss[-1]:.4f}\n\n"
        f"RL: {RL_STEPS} PPO steps\n"
        f"Final reward: {rl_rewards[-1]:+.3f}\n"
        f"Final KL: {rl_kl[-1]:.4f}\n\n"
        f"Model: d={D_MODEL}, h={N_HEADS}, L={N_LAYERS}\n"
        f"Device: {DEVICE}"
    )
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                     fontsize=12, verticalalignment="center",
                     fontfamily="monospace",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {OUTPUT_DIR}/training_curves.png")

    # Also save individual plots for README
    for name, data, color, ylabel in [
        ("sft_loss", sft_loss, "#2196F3", "Cross-Entropy Loss"),
        ("rm_loss", rm_loss, "#FF9800", "Comparison Loss"),
        ("rl_rewards", rl_rewards, "#4CAF50", "Reward"),
    ]:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(data, color=color, linewidth=2)
        ax2.set_xlabel("Step")
        ax2.set_ylabel(ylabel)
        ax2.set_title(name.replace("_", " ").title())
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=120)
        plt.close()

    print(f"  ✓ Saved individual plots: sft_loss.png, rm_loss.png, rl_rewards.png")


# ══════════════════════════════════════════════════════════════════════════════
# 12.  MAIN — ORCHESTRATE THE COMPLETE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Executes the full RLHF pipeline:

    ┌───────────┐    ┌─────────────┐    ┌────────────┐
    │ Phase 1   │    │ Phase 2     │    │ Phase 3    │
    │ SFT       │───▶│ Generate    │───▶│ Annotate   │
    │ Training  │    │ Candidates  │    │ Preferences│
    └───────────┘    └─────────────┘    └─────┬──────┘
                                              │
    ┌───────────┐    ┌─────────────┐    ┌─────▼──────┐
    │ Phase 6   │    │ Phase 5     │    │ Phase 4    │
    │ Evaluate  │◀───│ RL (PPO)    │◀───│ Train RM   │
    │ & Plot    │    │ Fine-Tune   │    │            │
    └───────────┘    └─────────────┘    └────────────┘
    """
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  Mini-Transformer RLHF Pipeline v2  (Full Methodology)          ║")
    print("║  SFT → Generate → Annotate → Train RM → RL → Evaluate          ║")
    print(f"║  Device: {str(DEVICE):<57s}║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    tok = CharTokeniser()
    print(f"\n  Vocab size : {tok.vocab_size}")
    print(f"  Train set  : {len(TRAIN_RECIPES)} recipes")
    print(f"  Eval set   : {len(EVAL_RECIPES)} recipes")

    # ── Phase 1: SFT ─────────────────────────────────────────────────────────
    sft_model = MiniTransformer(tok.vocab_size).to(DEVICE)
    params = sum(p.numel() for p in sft_model.parameters())
    print(f"  Parameters : {params:,}")

    sft_loss = train_sft(sft_model, tok)

    # ── Phase 2: Generate candidates ──────────────────────────────────────────
    all_candidates = generate_candidates(sft_model, tok)

    # ── Phase 3: Annotate / create comparison pairs ───────────────────────────
    comparison_pairs, annotation_log = annotate_candidates(all_candidates)

    # ── Phase 4: Train Reward Model ───────────────────────────────────────────
    rm = RewardModel(tok.vocab_size).to(DEVICE)
    rm_loss = train_reward_model(rm, tok, comparison_pairs)

    # ── Phase 5: RL Fine-Tuning ──────────────────────────────────────────────
    policy_model = copy.deepcopy(sft_model).to(DEVICE)
    ref_model    = copy.deepcopy(sft_model).to(DEVICE)
    ref_model.requires_grad_(False)    # freeze reference

    rl_rewards, rl_kl, rl_loss = rl_training_loop(
        policy_model, ref_model, rm, tok
    )

    # ── Phase 6: Evaluation ──────────────────────────────────────────────────
    eval_results = evaluate(sft_model, policy_model, rm, tok)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(sft_loss, rm_loss, rl_rewards, rl_kl, rl_loss)

    print("\n" + "═" * 72)
    print("  ✓  Full RLHF pipeline complete!")
    print(f"  ✓  Plots saved to ./{OUTPUT_DIR}/")
    print("═" * 72)


if __name__ == "__main__":
    main()
