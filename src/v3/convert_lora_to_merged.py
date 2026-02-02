#%% [markdown]
# # V3 LoRA → Merged Model 변환
#
# 이 스크립트는 저장된 LoRA 어댑터를 로드하고,
# base model과 merge하여 **전체 모델**로 저장합니다.
#
# 이렇게 하면 추론 시 PEFT가 필요 없어서 Kaggle (internet off)에서도 작동합니다.

#%%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#%%
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Paths
ADAPTER_DIR = Path("/content/drive/MyDrive/akkadian/v3/lora_adapter")
OUTPUT_DIR = Path("/content/drive/MyDrive/akkadian/v3/merged_model")

print(f"📁 LoRA adapter: {ADAPTER_DIR}")
print(f"📁 Output: {OUTPUT_DIR}")

#%%
# Load base model
print("\n🤖 Loading base model: google/byt5-large")
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-large")
print("   Base model loaded")

# Load tokenizer from adapter
print("\n🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
print(f"   Tokenizer vocab size: {len(tokenizer)}")

#%%
# Load LoRA adapter
print(f"\n🔧 Loading LoRA adapter from: {ADAPTER_DIR}")
model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
print("   LoRA adapter loaded")

# Merge weights
print("\n🔀 Merging adapter weights into base model...")
model = model.merge_and_unload()
print("   ✅ Merge complete!")

#%%
# Save merged model
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n💾 Saving merged model to: {OUTPUT_DIR}")
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print("   ✅ Saved!")

#%%
# Create ZIP archive
import shutil
zip_path = OUTPUT_DIR.parent / "akkadian_v3_merged"
shutil.make_archive(str(zip_path), 'zip', OUTPUT_DIR)
print(f"\n📦 Archive created: {zip_path}.zip")

print("\n" + "=" * 60)
print("✅ 변환 완료!")
print("=" * 60)
print(f"📁 Merged model: {OUTPUT_DIR}")
print(f"📦 Archive: {zip_path}.zip")
print("\n다음 단계:")
print("1. akkadian_v3_merged.zip 다운로드")
print("2. Kaggle Dataset으로 업로드")
print("3. 추론 코드에서 이 모델 사용 (PEFT 불필요!)")
print("=" * 60)
