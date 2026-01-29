# Akkadian V1 - Training & Inference Guide

## 📁 Files

| File | Purpose |
|------|---------|
| `akka_v1_train.py` | Training script (source) |
| `akka_v1_train.ipynb` | Training notebook (Kaggle upload) |
| `akka_v1_infer.py` | Inference script (source) |
| `akka_v1_infer.ipynb` | Inference notebook (Kaggle upload) |

## 🚀 Kaggle Workflow

### Step 1: Training

1. **Upload notebook** to Kaggle
   - Go to: [kaggle.com/code](https://kaggle.com/code) → New Notebook
   - Upload `akka_v1_train.ipynb`

2. **Add competition data**
   - Click "Add Data" → Search "deep-past-initiative-machine-translation"
   - Add the competition dataset

3. **Configure accelerator**
   - Settings → Accelerator → GPU T4 x2

4. **Run the notebook**
   - Click "Run All" or run cells sequentially
   - Training takes ~2-3 hours on T4 x2

5. **Save the output**
   - Click "Save Version" → "Always save output"
   - Model will be saved to `/kaggle/working/akkadian_v1/final/`

### Step 2: Upload Model to Kaggle

**Option A: As Kaggle Model**
1. Go to [kaggle.com/models](https://kaggle.com/models) → New Model
2. Name: `akkadian-byt5-v1`
3. Upload the contents of `/kaggle/working/akkadian_v1/final/`

**Option B: As Kaggle Dataset**
1. Download the `akkadian_v1/final/` folder from notebook output
2. Go to [kaggle.com/datasets](https://kaggle.com/datasets) → New Dataset
3. Upload the folder

### Step 3: Inference

1. **Upload notebook**
   - Upload `akka_v1_infer.ipynb` to Kaggle

2. **Add data sources**
   - Competition data (same as training)
   - Your trained model (from Step 2)

3. **Configure model path**
   Edit the Config in the notebook:
   ```python
   @dataclass
   class Config:
       # Option 1: Dataset name
       model_dataset_name: Optional[str] = "your-username/akkadian-byt5-v1"
       
       # Option 2: Direct path (if model is in a subfolder)
       model_path: Optional[Path] = None
   ```

4. **Run the notebook**
   - Output: `/kaggle/working/submission.csv`

5. **Submit**
   - Save the notebook version
   - Go to competition → Submit → Select this notebook

## ⚙️ Configuration Options

### Training (`akka_v1_train.py`)

```python
@dataclass
class Config:
    model_name: str = "google/byt5-base"  # or "google/byt5-small" for faster training
    
    # Training
    epochs: int = 5
    batch_size: int = 4           # per GPU
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    
    # Sequence lengths
    max_source_length: int = 256
    max_target_length: int = 256
    
    # Memory optimization
    gradient_checkpointing: bool = True  # reduces memory, slightly slower
    fp16: bool = True                    # faster on T4
```

### Inference (`akka_v1_infer.py`)

```python
@dataclass
class Config:
    batch_size: int = 8      # larger for inference
    num_beams: int = 4       # beam search
    fp16: bool = True        # faster inference
```

## 📊 Expected Results

| Metric | Approximate Score |
|--------|------------------|
| BLEU | 15-25 |
| chrF++ | 35-45 |
| Geo Mean | 25-35 |

*Actual scores depend on training time, data quality, and hyperparameters.*

## 🔧 Local Development

### Convert .py to .ipynb
```bash
uv run jupytext --to notebook src/akka_v1_train.py
uv run jupytext --to notebook src/akka_v1_infer.py
```

### Convert .ipynb back to .py
```bash
uv run jupytext --to py:percent src/akka_v1_train.ipynb
uv run jupytext --to py:percent src/akka_v1_infer.ipynb
```

### Sync both formats
```bash
uv run jupytext --sync src/akka_v1_train.py
uv run jupytext --sync src/akka_v1_infer.py
```

## 🐛 Troubleshooting

### "Model not found" error
- Check `CFG.model_dataset_name` matches your uploaded model name
- Verify model files include `config.json`

### Out of Memory
- Reduce `batch_size` to 2
- Enable `gradient_checkpointing = True`
- Use `byt5-small` instead of `byt5-base`

### Slow training
- Ensure GPU T4 x2 is selected
- Check `fp16 = True` is enabled

## 📚 References

- [Competition Page](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)
- [ByT5 Paper](https://arxiv.org/abs/2105.13626)
- [Hugging Face ByT5](https://huggingface.co/google/byt5-base)
