import os
import json
from datasets import load_dataset

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading WikiText-2...")
    wiki_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wiki_path = os.path.join(data_dir, "wikitext_test.txt")
    with open(wiki_path, "w", encoding="utf-8") as f:
        for item in wiki_dataset:
            text = item["text"].strip()
            if text:
                f.write(text + "\n")
    print(f"Saved WikiText-2 to {wiki_path}")
    
    print("Downloading HellaSwag...")
    hellaswag_dataset = load_dataset("hellaswag", split="validation")
    hellaswag_path = os.path.join(data_dir, "hellaswag_val.jsonl")
    with open(hellaswag_path, "w", encoding="utf-8") as f:
        for item in hellaswag_dataset:
            json.dump(item, f)
            f.write("\n")
    print(f"Saved HellaSwag to {hellaswag_path}")
    
    print("All datasets downloaded successfully!")

if __name__ == "__main__":
    main()
