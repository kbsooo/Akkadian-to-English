#%% [markdown]
# # Akkadian V4 Training: Full FT + OCR Noise Augmentation
#
# **Key Features:**
# - ByT5 (base or large) with Full Fine-tuning
# - OCR noise augmentation on RAW data
# - Original train.csv + published_texts.csv (AICC_translation)
# - oare_id-based train/val split to prevent leakage
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
# Download competition data
competition_path = kagglehub.competition_download('deep-past-initiative-machine-translation')
print(f'Competition data: {competition_path}')

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
from typing import List, Optional, Set

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sacrebleu.metrics import BLEU, CHRF
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
    
    # Paths
    data_dir: Path = None  # Set after download
    output_dir: Path = None
    
    # OCR Augmentation - applied BEFORE normalization
    augment_prob: float = 0.4
    use_published_texts: bool = False  # Reserved for DAPT phase
    
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
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    
    # Model-specific
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

CFG.data_dir = Path(competition_path)
CFG.output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"🚀 Akkadian V4: {CFG.model_size.upper()} + Full FT + OCR Augmentation")
print("=" * 60)
print(f"📁 Data: {CFG.data_dir}")
print(f"📁 Output: {CFG.output_dir}")
print(f"🤖 Model: {CFG.model_name}")
print(f"🎲 Augment prob: {CFG.augment_prob}")
print(f"📚 Use published_texts: {CFG.use_published_texts}")
print(f"🎮 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

set_seed(CFG.seed)

#%% [markdown]
# ## 2. Normalization (V2-identical)
#
# Applied to source AFTER augmentation, and to target always.

#%%
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
    '\u0161': 's', '\u0160': 'S',  # š
    '\u1e63': 's', '\u1e62': 'S',  # ṣ
    '\u1e6d': 't', '\u1e6c': 'T',  # ṭ
    '\u1e2b': 'h', '\u1e2a': 'H',  # ḫ
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
    """Normalize Akkadian transliteration to ASCII."""
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
# ## 3. OCR Noise Augmentation
#
# Key insight: Apply noise to RAW data, then normalize.
# Some noise (diacritics drop, hyphen drop) survives normalization.
# Quote/bracket variations are normalized away, but that's OK -
# the model sees variety during training which helps generalization.

#%%
_DIACRITICS_DROP = {
    '\u0161': 's', '\u0160': 'S',
    '\u1e63': 's', '\u1e62': 'S',
    '\u1e6d': 't', '\u1e6c': 'T',
    '\u1e2b': 'h', '\u1e2a': 'H',
    '\u0101': 'a', '\u0100': 'A',
    '\u0113': 'e', '\u0112': 'E',
    '\u012b': 'i', '\u012a': 'I',
    '\u016b': 'u', '\u016a': 'U',
}

_SUBSCRIPT_TO_NUM = {
    '\u2080': '0', '\u2081': '1', '\u2082': '2', '\u2083': '3', '\u2084': '4',
    '\u2085': '5', '\u2086': '6', '\u2087': '7', '\u2088': '8', '\u2089': '9',
}
_NUM_TO_SUBSCRIPT = {v: k for k, v in _SUBSCRIPT_TO_NUM.items()}


def _drop_diacritics(text: str, prob: float = 0.4) -> str:
    """Randomly drop diacritics (š→s, etc.)."""
    result = []
    for char in text:
        if char in _DIACRITICS_DROP and random.random() < prob:
            result.append(_DIACRITICS_DROP[char])
        else:
            result.append(char)
    return ''.join(result)


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


def _drop_some_hyphens(text: str, prob: float = 0.15) -> str:
    """Randomly drop some hyphens (not all)."""
    words = text.split()
    result = []
    for word in words:
        if '-' in word and random.random() < prob:
            word = word.replace('-', '')
        result.append(word)
    return ' '.join(result)


def _protect_and_restore_tokens(text: str, tokens: List[str]) -> tuple:
    """Protect special tokens from modification."""
    protected = {}
    counter = 0
    for token in tokens:
        while token in text:
            placeholder = f"__P{counter}__"
            text = text.replace(token, placeholder, 1)
            protected[placeholder] = token
            counter += 1
    return text, protected


def _restore_tokens(text: str, protected: dict) -> str:
    for placeholder, token in protected.items():
        text = text.replace(placeholder, token)
    return text


def apply_ocr_noise(text: str, prob: float = 0.4) -> str:
    """
    Apply OCR-like noise to RAW transliteration.
    
    Effective noise types (survive normalization):
    - Diacritics drop: š→s (most impactful)
    - Subscript variation: ₄→4
    - Hyphen drop: qa-ti→qati
    """
    if not text or random.random() > prob:
        return text
    
    # Protect special tokens
    text, protected = _protect_and_restore_tokens(
        text, ['<gap>', '<unk>', '<GAP>', '<UNK>']
    )
    
    # Apply noise functions (these survive normalization)
    noise_funcs = [
        lambda t: _drop_diacritics(t, prob=0.35),
        lambda t: _vary_subscripts(t, prob=0.25),
        lambda t: _drop_some_hyphens(t, prob=0.12),
    ]
    
    n_funcs = random.randint(1, len(noise_funcs))
    for func in random.sample(noise_funcs, n_funcs):
        text = func(text)
    
    text = _restore_tokens(text, protected)
    return text


# Test
print("\n📝 OCR Augmentation Examples (RAW → Noisy):")
test_text = "šum-ma a-wi-lum ṣa-bi-tam ḫu-bu-ul-li-šu₄"
for i in range(5):
    noisy = apply_ocr_noise(test_text, prob=1.0)
    normalized = normalize_transliteration(noisy)
    print(f"   [{i}] {normalized}")

#%% [markdown]
# ## 4. Load Data with oare_id-based Split

#%%
def find_file(data_dir: Path, filename: str) -> Path:
    """Find file in data directory or subdirectories."""
    if (data_dir / filename).exists():
        return data_dir / filename
    for p in data_dir.glob(f"**/{filename}"):
        return p
    raise FileNotFoundError(f"{filename} not found in {data_dir}")


def load_raw_data():
    """Load RAW data with oare_id-based split to prevent leakage."""
    
    # 1. Load train.csv
    train_path = find_file(CFG.data_dir, "train.csv")
    train_df = pd.read_csv(train_path)
    print(f"   train.csv: {len(train_df)} rows")
    
    # Standardize columns
    train_df = train_df.rename(columns={
        'transliteration': 'src_raw',
        'translation': 'tgt_raw'
    })
    train_df['source'] = 'train'
    
    # 2. Load published_texts.csv if enabled
    if CFG.use_published_texts:
        try:
            pub_path = find_file(CFG.data_dir, "published_texts.csv")
            pub_df = pd.read_csv(pub_path)
            print(f"   published_texts.csv: {len(pub_df)} rows total")
            
            # Use correct column names!
            # transliteration (or transliteration_orig) → src_raw
            # AICC_translation → tgt_raw
            pub_df = pub_df.rename(columns={
                'transliteration': 'src_raw',
                'AICC_translation': 'tgt_raw'
            })
            
            # Keep only rows with both src and tgt
            pub_df = pub_df.dropna(subset=['src_raw', 'tgt_raw'])
            pub_df = pub_df[pub_df['tgt_raw'].str.strip().str.len() > 0]
            
            print(f"   published_texts with AICC_translation: {len(pub_df)} rows")
            
            if len(pub_df) > 0:
                pub_df['source'] = 'published'
                train_df = pd.concat([train_df, pub_df[['oare_id', 'src_raw', 'tgt_raw', 'source']]], ignore_index=True)
        except FileNotFoundError:
            print("   ⚠️ published_texts.csv not found, skipping")
    
    # Drop NaN
    train_df = train_df.dropna(subset=['src_raw', 'tgt_raw']).reset_index(drop=True)
    
    return train_df


def split_by_oare_id(df: pd.DataFrame, val_size: float, seed: int):
    """
    Split by oare_id to prevent train/val leakage.
    All rows with same oare_id go to same split.
    """
    unique_ids = df['oare_id'].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_ids)
    
    n_val = int(len(unique_ids) * val_size)
    val_ids = set(unique_ids[:n_val])
    train_ids = set(unique_ids[n_val:])
    
    train_df = df[df['oare_id'].isin(train_ids)].reset_index(drop=True)
    val_df = df[df['oare_id'].isin(val_ids)].reset_index(drop=True)
    
    return train_df, val_df


print("📖 Loading raw data...")
raw_df = load_raw_data()
print(f"   Total: {len(raw_df)} rows")

# Split by oare_id
print(f"\n🔀 Splitting by oare_id (val_size={CFG.val_size})...")
train_df, val_df = split_by_oare_id(raw_df, CFG.val_size, CFG.seed)
print(f"   Train: {len(train_df)} rows ({train_df['oare_id'].nunique()} documents)")
print(f"   Val: {len(val_df)} rows ({val_df['oare_id'].nunique()} documents)")

# Verify no leakage
train_ids = set(train_df['oare_id'].unique())
val_ids = set(val_df['oare_id'].unique())
assert len(train_ids & val_ids) == 0, "Train/val leakage detected!"
print("   ✅ No oare_id overlap (no leakage)")

#%% [markdown]
# ## 5. Prepare Data with Augmentation

#%%
def prepare_data(df: pd.DataFrame, augment: bool = False) -> pd.DataFrame:
    """
    Prepare data:
    1. (Optional) Apply OCR noise to RAW source
    2. Normalize both source and target
    """
    prepared = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing"):
        src_raw = row['src_raw']
        tgt_raw = row['tgt_raw']
        
        # Apply noise to RAW source
        if augment:
            src_noisy = apply_ocr_noise(src_raw, prob=CFG.augment_prob)
        else:
            src_noisy = src_raw
        
        # Normalize
        src = normalize_transliteration(src_noisy)
        tgt = normalize_translation(tgt_raw)
        
        if src and tgt:
            prepared.append({'src': src, 'tgt': tgt})
    
    return pd.DataFrame(prepared)


print("\n🔧 Preparing training data (with augmentation)...")
train_prepared = prepare_data(train_df, augment=True)
print(f"   Train: {len(train_prepared)} samples")

print("\n🔧 Preparing validation data (no augmentation)...")
val_prepared = prepare_data(val_df, augment=False)
print(f"   Val: {len(val_prepared)} samples")

print(f"\n📝 Sample:")
print(f"   src: {train_prepared.iloc[0]['src'][:80]}...")
print(f"   tgt: {train_prepared.iloc[0]['tgt'][:80]}...")

#%% [markdown]
# ## 6. Model Setup

#%%
print(f"\n🤖 Loading model: {CFG.model_name}")

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(CFG.model_name)

print(f"   Tokenizer: {len(tokenizer)}, Model: {model.config.vocab_size}")
print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")

assert len(tokenizer) == model.config.vocab_size, "Vocab mismatch!"

if CFG.gradient_checkpointing:
    model.gradient_checkpointing_enable()
    print("   ✅ Gradient checkpointing enabled")

#%% [markdown]
# ## 7. Tokenization

#%%
def tokenize_fn(examples):
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


print("\n🔤 Tokenizing...")
train_ds = Dataset.from_pandas(train_prepared[["src", "tgt"]])
train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"])

val_ds = Dataset.from_pandas(val_prepared[["src", "tgt"]])
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["src", "tgt"])

print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")

#%% [markdown]
# ## 8. Metrics & Callback

#%%
def build_compute_metrics(tokenizer):
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


class LogCallback(TrainerCallback):
    def __init__(self):
        self.epoch = 0
        self.losses = []
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch = int(state.epoch) if state.epoch else 0
        self.losses = []
        print(f"\n{'='*60}\n📊 Epoch {self.epoch + 1}/{args.num_train_epochs}\n{'='*60}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.losses:
            print(f"\n📉 Train Loss: {sum(self.losses)/len(self.losses):.4f}")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n{'─'*40}\n📈 Validation\n{'─'*40}")
            print(f"   BLEU: {metrics.get('eval_bleu', 0):.2f}")
            print(f"   chrF: {metrics.get('eval_chrf', 0):.2f}")
            print(f"   Geo:  {metrics.get('eval_geo_mean', 0):.2f}\n{'─'*40}")

#%% [markdown]
# ## 9. Training

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
    fp16=CFG.fp16,
    bf16=CFG.bf16,
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
    report_to="none",
    seed=CFG.seed,
)

try:
    training_args = Seq2SeqTrainingArguments(**training_kwargs)
except TypeError:
    training_kwargs["eval_strategy"] = training_kwargs.pop("evaluation_strategy")
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=build_compute_metrics(tokenizer),
    callbacks=[LogCallback()],
)

print(f"\n🏋️ Training: {CFG.epochs} epochs, batch {CFG.batch_size}x{CFG.gradient_accumulation_steps}")
trainer.train()

#%% [markdown]
# ## 10. Save & Archive

#%%
model_dir = CFG.output_dir / "model"
trainer.save_model(str(model_dir))
tokenizer.save_pretrained(str(model_dir))
print(f"\n💾 Saved: {model_dir}")

results = trainer.evaluate()
print(f"\n📈 Final: BLEU={results.get('eval_bleu',0):.2f}, chrF={results.get('eval_chrf',0):.2f}, Geo={results.get('eval_geo_mean',0):.2f}")

import shutil
zip_path = CFG.output_dir / f"akkadian_v4_{CFG.model_size}"
shutil.make_archive(str(zip_path), 'zip', model_dir)
print(f"📦 Archive: {zip_path}.zip")

print(f"\n{'='*60}\n✅ V4-{CFG.model_size.upper()} Complete!\n{'='*60}")
