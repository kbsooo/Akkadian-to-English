#%% [markdown]
# # Akkadian V4 Training: Full FT + OCR Noise Augmentation
#
# **Key Features:**
# - ByT5 (base or large) with Full Fine-tuning
# - OCR noise augmentation on RAW data (not pre-normalized)
# - Original train.csv + published_texts.csv for more diversity
#
# **Environment**: Google Colab with A100 GPU
#
# **Output**: Saved to Google Drive `/content/drive/MyDrive/akkadian/v4`

#%% [markdown]
# ## 0. Setup: Mount Drive & Download Data

#%%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#%%
# Kaggle Hub login and data download
import kagglehub
kagglehub.login()

#%%
# Download competition data (contains raw train.csv)
competition_path = kagglehub.dataset_download('deep-past-initiative-machine-translation')
print(f'Competition data: {competition_path}')

# Download published_texts for augmentation
published_path = kagglehub.dataset_download('kbsooo/akkadian-v2-data')
print(f'V2 data (published_texts): {published_path}')

#%% [markdown]
# ## 1. Imports & Configuration

#%%
from __future__ import annotations

import os
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sacrebleu.metrics import BLEU, CHRF
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    set_seed,
)

#%%
@dataclass
class Config:
    """Training configuration for V4: Full FT + OCR Augmentation."""
    
    # Model selection: "base" or "large"
    model_size: str = "base"  # Change to "large" for v4-large
    
    # Paths (set after data download)
    competition_dir: Path = None
    v2_data_dir: Path = None
    output_dir: Path = None
    
    # OCR Augmentation
    augment_prob: float = 0.4  # Probability of applying noise
    use_published_texts: bool = True  # Include published_texts.csv
    
    # Training
    seed: int = 42
    val_size: float = 0.1
    max_source_length: int = 512
    max_target_length: int = 512
    epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Hardware
    fp16: bool = False  # ByT5 is unstable with FP16
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    
    # Model-specific settings (set in __post_init__)
    model_name: str = field(init=False)
    batch_size: int = field(init=False)
    gradient_accumulation_steps: int = field(init=False)
    learning_rate: float = field(init=False)
    
    def __post_init__(self):
        if self.model_size == "base":
            self.model_name = "google/byt5-base"
            self.batch_size = 4
            self.gradient_accumulation_steps = 4
            self.learning_rate = 1e-4
            self.output_dir = Path("/content/drive/MyDrive/akkadian/v4-base")
        else:
            self.model_name = "google/byt5-large"
            self.batch_size = 2
            self.gradient_accumulation_steps = 8
            self.learning_rate = 5e-5
            self.output_dir = Path("/content/drive/MyDrive/akkadian/v4-large")


# ============================================
# ⚠️ CHANGE THIS FOR v4-large
# ============================================
CFG = Config(model_size="base")  # "base" or "large"

# Set data directories
CFG.competition_dir = Path(competition_path)
CFG.v2_data_dir = Path(published_path)
CFG.output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"🚀 Akkadian V4: {CFG.model_size.upper()} + Full FT + OCR Augmentation")
print("=" * 60)
print(f"📁 Competition data: {CFG.competition_dir}")
print(f"📁 V2 data: {CFG.v2_data_dir}")
print(f"📁 Output: {CFG.output_dir}")
print(f"🤖 Model: {CFG.model_name}")
print(f"🎲 Augment prob: {CFG.augment_prob}")
print(f"📚 Use published_texts: {CFG.use_published_texts}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

set_seed(CFG.seed)

#%% [markdown]
# ## 2. OCR Noise Augmentation (for RAW data)

#%%
# ==============================================================================
# OCR Noise Augmentation Functions
# Applied to RAW transliteration (with diacritics) for maximum effect
# ==============================================================================

# Diacritics that can be dropped (simulating OCR errors)
_DIACRITICS_DROP = {
    '\u0161': 's', '\u0160': 'S',  # š, Š
    '\u1e63': 's', '\u1e62': 'S',  # ṣ, Ṣ
    '\u1e6d': 't', '\u1e6c': 'T',  # ṭ, Ṭ
    '\u1e2b': 'h', '\u1e2a': 'H',  # ḫ, Ḫ
    '\u0101': 'a', '\u0100': 'A',  # ā, Ā
    '\u0113': 'e', '\u0112': 'E',  # ē, Ē
    '\u012b': 'i', '\u012a': 'I',  # ī, Ī
    '\u016b': 'u', '\u016a': 'U',  # ū, Ū
}

# Quote variations (ASCII and Unicode)
_QUOTE_VARIATIONS = [
    ('"', '\u201e'),  # " → „
    ('"', '\u201c'),  # " → "
    ('"', '\u201d'),  # " → "
    ("'", '\u2018'),  # ' → '
    ("'", '\u2019'),  # ' → '
]

# Subscript ↔ number
_SUBSCRIPT_TO_NUM = {
    '\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3', '\u2084': '4',
    '\u2085': '5', '\u2086': '6', '\u2087': '7', '\u2088': '8', '\u2089': '9',
}
_NUM_TO_SUBSCRIPT = {v: k for k, v in _SUBSCRIPT_TO_NUM.items()}


def _drop_diacritics(text: str, prob: float = 0.4) -> str:
    """Randomly drop diacritics (simulating OCR/ASCII degradation)."""
    result = []
    for char in text:
        if char in _DIACRITICS_DROP and random.random() < prob:
            result.append(_DIACRITICS_DROP[char])
        else:
            result.append(char)
    return ''.join(result)


def _vary_quotes(text: str) -> str:
    """Randomly swap quote styles."""
    for orig, alt in _QUOTE_VARIATIONS:
        if random.random() < 0.3:
            if random.random() < 0.5:
                text = text.replace(orig, alt)
            else:
                text = text.replace(alt, orig)
    return text


def _vary_subscripts(text: str, prob: float = 0.3) -> str:
    """Randomly convert subscripts ↔ numbers."""
    result = []
    for char in text:
        if char in _SUBSCRIPT_TO_NUM and random.random() < prob:
            result.append(_SUBSCRIPT_TO_NUM[char])
        elif char in _NUM_TO_SUBSCRIPT and random.random() < prob * 0.3:
            result.append(_NUM_TO_SUBSCRIPT[char])
        else:
            result.append(char)
    return ''.join(result)


def _drop_hyphens(text: str, prob: float = 0.1) -> str:
    """Randomly drop some hyphens (but not all)."""
    if random.random() > prob:
        return text
    words = text.split()
    result = []
    for word in words:
        if random.random() < 0.3:
            word = word.replace('-', '')
        result.append(word)
    return ' '.join(result)


def _vary_brackets(text: str) -> str:
    """Vary bracket styles (philological notation)."""
    if random.random() < 0.2:
        text = text.replace('[', '\u2308').replace(']', '\u2309')  # ⌈ ⌉
    if random.random() < 0.2:
        text = text.replace('[', '\u230a').replace(']', '\u230b')  # ⌊ ⌋
    return text


def _protect_special_tokens(text: str) -> tuple:
    """Extract and protect special tokens like <gap>, <unk>."""
    protected = {}
    counter = 0
    for token in ['<gap>', '<unk>', '<GAP>', '<UNK>']:
        while token in text:
            placeholder = f"__PROTECTED_{counter}__"
            text = text.replace(token, placeholder, 1)
            protected[placeholder] = token
            counter += 1
    return text, protected


def _restore_special_tokens(text: str, protected: dict) -> str:
    """Restore protected special tokens."""
    for placeholder, token in protected.items():
        text = text.replace(placeholder, token)
    return text


def apply_ocr_noise(text: str, prob: float = 0.4) -> str:
    """
    Apply random OCR-like noise to RAW transliteration.
    
    Protects special tokens (<gap>, <unk>) from corruption.
    Only applied during TRAINING, not inference.
    """
    if not text or random.random() > prob:
        return text
    
    # Protect special tokens
    text, protected = _protect_special_tokens(text)
    
    # Apply 1-3 random noise functions
    noise_funcs = [
        lambda t: _drop_diacritics(t, prob=0.3),
        _vary_quotes,
        lambda t: _vary_subscripts(t, prob=0.2),
        lambda t: _drop_hyphens(t, prob=0.1),
        _vary_brackets,
    ]
    
    n_funcs = random.randint(1, min(3, len(noise_funcs)))
    selected_funcs = random.sample(noise_funcs, n_funcs)
    
    for func in selected_funcs:
        text = func(text)
    
    # Restore special tokens
    text = _restore_special_tokens(text, protected)
    
    return text


# Test augmentation
print("\n📝 OCR Augmentation Examples:")
test_text = "šum-ma a-wi-lum ṣa-bi-tam i-na ḫu-bu-ul-li-šu <gap>"
for i in range(5):
    augmented = apply_ocr_noise(test_text, prob=1.0)
    print(f"   [{i}] {augmented}")

#%% [markdown]
# ## 3. Normalization (for consistent output)

#%%
# ==============================================================================
# Normalization (V2-identical) - applied to BOTH train source and target
# ==============================================================================

_VOWEL_MAP = {
    '\u00e0': 'a', '\u00e1': 'a', '\u00e2': 'a', '\u0101': 'a', '\u00e4': 'a',
    '\u00c0': 'A', '\u00c1': 'A', '\u00c2': 'A', '\u0100': 'A', '\u00c4': 'A',
    '\u00e8': 'e', '\u00e9': 'e', '\u00ea': 'e', '\u0113': 'e', '\u00eb': 'e',
    '\u00c8': 'E', '\u00c9': 'E', '\u00ca': 'E', '\u0112': 'E', '\u00cb': 'E',
    '\u00ec': 'i', '\u00ed': 'i', '\u00ee': 'i', '\u012b': 'i', '\u00ef': 'i',
    '\u00cc': 'I', '\u00cd': 'I', '\u00ce': 'I', '\u012a': 'I', '\u00cf': 'I',
    '\u00f2': 'o', '\u00f3': 'o', '\u00f4': 'o', '\u014d': 'o', '\u00f6': 'o',
    '\u00d2': 'O', '\u00d3': 'O', '\u00d4': 'O', '\u014c': 'O', '\u00d6': 'O',
    '\u00f9': 'u', '\u00fa': 'u', '\u00fb': 'u', '\u016b': 'u', '\u00fc': 'u',
    '\u00d9': 'U', '\u00da': 'U', '\u00db': 'U', '\u016a': 'U', '\u00dc': 'U',
}

_CONSONANT_MAP = {
    '\u0161': 's', '\u0160': 'S',
    '\u1e63': 's', '\u1e62': 'S',
    '\u1e6d': 't', '\u1e6c': 'T',
    '\u1e2b': 'h', '\u1e2a': 'H',
}

_OCR_MAP = {
    '\u201e': '"', '\u201c': '"', '\u201d': '"',
    '\u2018': "'", '\u2019': "'", '\u201a': "'",
    '\u02be': "'", '\u02bf': "'",
    '\u2308': '[', '\u2309': ']', '\u230a': '[', '\u230b': ']',
}

_SUBSCRIPT_MAP = str.maketrans({
    '\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3', '\u2084': '4',
    '\u2085': '5', '\u2086': '6', '\u2087': '7', '\u2088': '8', '\u2089': '9',
    '\u2093': 'x',
})

_FULL_MAP = str.maketrans({**_VOWEL_MAP, **_CONSONANT_MAP, **_OCR_MAP})


def normalize_transliteration(text) -> str:
    """Normalize Akkadian transliteration to ASCII (V2-identical)."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_FULL_MAP)
    text = text.translate(_SUBSCRIPT_MAP)
    text = text.replace('\u2026', ' <gap> ')
    text = re.sub(r'\.\.\.+', ' <gap> ', text)
    text = re.sub(r'\[([^\]]*)\]', ' <gap> ', text)
    text = re.sub(r'\bx\b', ' <unk> ', text, flags=re.IGNORECASE)
    text = re.sub(r'[!?/]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_translation(text) -> str:
    """Normalize English translation."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#%% [markdown]
# ## 4. Load and Prepare Data

#%%
def load_raw_data():
    """Load RAW data from competition + published_texts."""
    
    # 1. Load original train.csv (RAW, with diacritics)
    train_path = CFG.competition_dir / "train.csv"
    if not train_path.exists():
        # Try alternative paths
        for p in CFG.competition_dir.glob("**/train.csv"):
            train_path = p
            break
    
    train_df = pd.read_csv(train_path)
    print(f"   Raw train.csv: {len(train_df)} rows")
    
    # Rename columns for consistency
    train_df = train_df.rename(columns={
        'transliteration': 'src_raw',
        'translation': 'tgt_raw'
    })
    train_df['source'] = 'train'
    
    # 2. Load published_texts.csv if enabled
    if CFG.use_published_texts:
        pub_path = CFG.competition_dir / "published_texts.csv"
        if not pub_path.exists():
            for p in CFG.competition_dir.glob("**/published_texts.csv"):
                pub_path = p
                break
        
        if pub_path.exists():
            pub_df = pd.read_csv(pub_path)
            print(f"   published_texts.csv: {len(pub_df)} rows")
            
            # Only use rows with both src and tgt
            pub_df = pub_df.rename(columns={
                'transliteration': 'src_raw',
                'translation': 'tgt_raw'
            })
            pub_df = pub_df.dropna(subset=['src_raw', 'tgt_raw'])
            pub_df['source'] = 'published'
            
            print(f"   published_texts with translations: {len(pub_df)} rows")
            
            # Combine
            train_df = pd.concat([train_df, pub_df], ignore_index=True)
    
    # Drop NaN
    train_df = train_df.dropna(subset=['src_raw', 'tgt_raw']).reset_index(drop=True)
    
    return train_df


print("📖 Loading raw data...")
raw_df = load_raw_data()
print(f"   Total: {len(raw_df)} rows")

# Show sample
print(f"\n📝 Sample (RAW):")
print(f"   src: {raw_df.iloc[0]['src_raw'][:80]}...")
print(f"   tgt: {raw_df.iloc[0]['tgt_raw'][:80]}...")

#%%
# Split into train/val
print(f"\n🔀 Splitting into train/val (val_size={CFG.val_size})...")
train_df, val_df = train_test_split(
    raw_df, test_size=CFG.val_size, random_state=CFG.seed
)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print(f"   Train: {len(train_df)} rows")
print(f"   Val: {len(val_df)} rows")

#%% [markdown]
# ## 5. Data Preparation with Augmentation

#%%
def prepare_training_data(df, augment=False):
    """
    Prepare data for training:
    1. (Optional) Apply OCR noise to source
    2. Normalize both source and target
    """
    prepared = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing"):
        src_raw = row['src_raw']
        tgt_raw = row['tgt_raw']
        
        # Apply OCR noise to RAW source (before normalization)
        if augment:
            src_noisy = apply_ocr_noise(src_raw, prob=CFG.augment_prob)
        else:
            src_noisy = src_raw
        
        # Normalize both
        src = normalize_transliteration(src_noisy)
        tgt = normalize_translation(tgt_raw)
        
        if src and tgt:  # Skip empty
            prepared.append({'src': src, 'tgt': tgt})
    
    return pd.DataFrame(prepared)


print("\n🔧 Preparing training data (with augmentation)...")
train_prepared = prepare_training_data(train_df, augment=True)
print(f"   Prepared train: {len(train_prepared)} rows")

print("\n🔧 Preparing validation data (no augmentation)...")
val_prepared = prepare_training_data(val_df, augment=False)
print(f"   Prepared val: {len(val_prepared)} rows")

print(f"\n📝 Sample (after augmentation + normalization):")
print(f"   src: {train_prepared.iloc[0]['src'][:80]}...")
print(f"   tgt: {train_prepared.iloc[0]['tgt'][:80]}...")

#%% [markdown]
# ## 6. Model Setup

#%%
print(f"\n🤖 Loading model: {CFG.model_name}")
print("   This may take a few minutes...")

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name)

print(f"   Tokenizer vocab: {len(tokenizer)}")
print(f"   Model vocab: {model.config.vocab_size}")
print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")

# Verify vocab match
assert len(tokenizer) == model.config.vocab_size, \
    f"Vocab mismatch! Tokenizer: {len(tokenizer)}, Model: {model.config.vocab_size}"

if CFG.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    print("   ✅ Gradient checkpointing enabled")

#%% [markdown]
# ## 7. Tokenization

#%%
def tokenize_fn(examples):
    """Tokenize source and target."""
    model_inputs = tokenizer(
        examples["src"],
        max_length=CFG.max_source_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        examples["tgt"],
        max_length=CFG.max_target_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("\n🔤 Tokenizing datasets...")

train_ds = Dataset.from_pandas(train_prepared[["src", "tgt"]])
train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"])

val_ds = Dataset.from_pandas(val_prepared[["src", "tgt"]])
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"])

print(f"   Train: {len(train_ds)} samples")
print(f"   Val: {len(val_ds)} samples")

#%% [markdown]
# ## 8. Metrics

#%%
def build_compute_metrics(tokenizer):
    """Build metrics computation function."""
    bleu = BLEU()
    chrf = CHRF(word_order=2)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]
        
        bleu_score = bleu.corpus_score(decoded_preds, decoded_labels).score
        chrf_score = chrf.corpus_score(decoded_preds, decoded_labels).score
        geo_mean = np.sqrt(bleu_score * chrf_score) if bleu_score > 0 and chrf_score > 0 else 0.0
        
        return {"bleu": bleu_score, "chrf": chrf_score, "geo_mean": geo_mean}
    
    return compute_metrics

#%% [markdown]
# ## 9. Logging Callback

#%%
class TqdmLoggingCallback(TrainerCallback):
    """Enhanced logging with clear metrics display."""
    
    def __init__(self):
        self.current_epoch = 0
        self.train_loss = []
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.current_epoch = int(state.epoch) if state.epoch else 0
        self.train_loss = []
        print(f"\n{'='*60}")
        print(f"📊 Epoch {self.current_epoch + 1}/{args.num_train_epochs}")
        print(f"{'='*60}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_loss.append(logs["loss"])
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.train_loss:
            avg_loss = sum(self.train_loss) / len(self.train_loss)
            print(f"\n📉 Epoch {self.current_epoch + 1} Train Loss: {avg_loss:.4f}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n{'─'*40}")
            print(f"📈 Validation Results (Epoch {self.current_epoch + 1})")
            print(f"{'─'*40}")
            print(f"   Loss:     {metrics.get('eval_loss', 0):.4f}")
            print(f"   BLEU:     {metrics.get('eval_bleu', 0):.2f}")
            print(f"   chrF++:   {metrics.get('eval_chrf', 0):.2f}")
            print(f"   Geo Mean: {metrics.get('eval_geo_mean', 0):.2f}")
            print(f"{'─'*40}")
    
    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'='*60}")
        print("🎉 Training Complete!")
        print(f"   Total steps: {state.global_step:,}")
        if state.best_metric:
            print(f"   Best metric: {state.best_metric:.2f}")
        print(f"{'='*60}")

#%% [markdown]
# ## 10. Training

#%%
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

training_kwargs = dict(
    output_dir=str(CFG.output_dir / "checkpoints"),
    num_train_epochs=CFG.epochs,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=CFG.batch_size * 2,
    gradient_accumulation_steps=CFG.gradient_accumulation_steps,
    learning_rate=CFG.learning_rate,
    weight_decay=CFG.weight_decay,
    warmup_ratio=CFG.warmup_ratio,
    max_grad_norm=CFG.max_grad_norm,
    fp16=CFG.fp16 and torch.cuda.is_available(),
    bf16=CFG.bf16 and torch.cuda.is_available(),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_geo_mean",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=CFG.max_target_length,
    dataloader_num_workers=CFG.dataloader_num_workers,
    logging_steps=50,
    logging_first_step=True,
    report_to="none",
    seed=CFG.seed,
    disable_tqdm=False,
)

try:
    training_args = Seq2SeqTrainingArguments(**training_kwargs)
except TypeError:
    training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

#%%
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=build_compute_metrics(tokenizer),
    callbacks=[TqdmLoggingCallback()],
)

#%%
print("\n🏋️ Starting training...")
print(f"   Model: {CFG.model_size.upper()}")
print(f"   Data: {len(train_ds)} train, {len(val_ds)} val")
print(f"   Epochs: {CFG.epochs}")
print(f"   Batch: {CFG.batch_size} x {CFG.gradient_accumulation_steps} = {CFG.batch_size * CFG.gradient_accumulation_steps}")
print(f"   LR: {CFG.learning_rate}")
print(f"   Augment: {CFG.augment_prob}")
print()

trainer.train()

#%% [markdown]
# ## 11. Save Model

#%%
model_dir = CFG.output_dir / "model"
print(f"\n💾 Saving model to: {model_dir}")
trainer.save_model(str(model_dir))
tokenizer.save_pretrained(str(model_dir))

#%%
print("\n📈 Final Evaluation:")
results = trainer.evaluate()
print(f"   BLEU:     {results.get('eval_bleu', 0):.2f}")
print(f"   chrF++:   {results.get('eval_chrf', 0):.2f}")
print(f"   Geo Mean: {results.get('eval_geo_mean', 0):.2f}")

#%% [markdown]
# ## 12. Create Archive

#%%
import shutil

zip_path = CFG.output_dir / f"akkadian_v4_{CFG.model_size}"
shutil.make_archive(str(zip_path), 'zip', model_dir)
print(f"\n📦 Model archived: {zip_path}.zip")

print("\n" + "=" * 60)
print(f"✅ V4-{CFG.model_size.upper()} Training Complete!")
print("=" * 60)
print(f"📁 Model: {model_dir}")
print(f"📦 Archive: {zip_path}.zip")
print("\nNext steps:")
print("1. Download the archive from Google Drive")
print("2. Upload to Kaggle as a dataset for inference")
print("=" * 60)
