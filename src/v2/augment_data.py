"""
Augment training data with Sentences_Oare + published_texts.

This script:
1. Groups sentence translations by document (text_uuid)
2. Joins with published_texts to get full transliteration
3. Creates additional training pairs that aren't in train.csv
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent))
from normalize import normalize_transliteration, normalize_translation


def main():
    data_dir = Path("data")
    output_dir = Path("data/v2")
    
    print("Loading data...")
    sent = pd.read_csv(data_dir / "Sentences_Oare_FirstWord_LinNum.csv")
    pub = pd.read_csv(data_dir / "published_texts.csv")
    train = pd.read_csv(data_dir / "train.csv")
    
    print(f"  Sentences: {len(sent)}")
    print(f"  published_texts: {len(pub)}")
    print(f"  train.csv: {len(train)}")
    
    # 1. Aggregate sentence translations per document
    print("\nAggregating sentences by document...")
    sent = sent[sent['translation'].notna()]
    sent = sent.sort_values(['text_uuid', 'sentence_obj_in_text'])
    
    doc_trans = sent.groupby('text_uuid').agg({
        'translation': lambda x: ' '.join(x.astype(str))
    }).reset_index()
    doc_trans.columns = ['text_uuid', 'full_translation']
    print(f"  Documents with translations: {len(doc_trans)}")
    
    # 2. Join with published_texts
    print("\nJoining with published_texts...")
    merged = doc_trans.merge(
        pub[['oare_id', 'transliteration']], 
        left_on='text_uuid', 
        right_on='oare_id', 
        how='inner'
    )
    print(f"  Matched documents: {len(merged)}")
    
    # 3. Remove overlap with train.csv
    train_ids = set(train['oare_id'])
    new_data = merged[~merged['oare_id'].isin(train_ids)].copy()
    print(f"  New documents (not in train): {len(new_data)}")
    
    # 4. Normalize
    print("\nNormalizing...")
    new_data['src'] = new_data['transliteration'].apply(normalize_transliteration)
    new_data['tgt'] = new_data['full_translation'].apply(normalize_translation)
    new_data = new_data[['oare_id', 'src', 'tgt']]
    
    # 5. Filter quality
    mask = (new_data['src'].str.len() > 10) & (new_data['tgt'].str.len() > 10)
    new_data = new_data[mask]
    print(f"  After quality filter: {len(new_data)}")
    
    # 6. Load existing v2 data and merge
    print("\nMerging with existing v2 data...")
    v2_train = pd.read_csv(output_dir / "v2_train.csv")
    v2_val = pd.read_csv(output_dir / "v2_val.csv")
    
    print(f"  Existing v2_train: {len(v2_train)}")
    print(f"  Existing v2_val: {len(v2_val)}")
    
    # Add new data to train
    combined_train = pd.concat([v2_train, new_data], ignore_index=True)
    combined_train = combined_train.drop_duplicates(subset=['src', 'tgt'])
    
    print(f"\n  Combined v2_train: {len(combined_train)}")
    
    # 7. Save
    combined_train.to_csv(output_dir / "v2_train_augmented.csv", index=False)
    print(f"\n✅ Saved: {output_dir / 'v2_train_augmented.csv'}")
    
    # Sample
    if len(new_data) > 0:
        print("\n📝 Sample new data:")
        sample = new_data.iloc[0]
        print(f"  src: {sample['src'][:100]}...")
        print(f"  tgt: {sample['tgt'][:100]}...")


if __name__ == "__main__":
    main()
