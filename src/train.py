"""Fine-tune a small seq2seq model to classify a corporate-action notice and extract its key fields.

Usage (project root):
    python -m src.train                                   # flan-t5-base, full fine-tune (fits a 12 GB GPU)
    python -m src.train --model google/flan-t5-large --lora   # bigger model with LoRA adapters
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

from src.schema import PREFIX


class NoticeDataset(Dataset):
    def __init__(self, path, tok, max_src=512, max_tgt=128):
        with open(path, encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f]
        self.tok, self.max_src, self.max_tgt = tok, max_src, max_tgt

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(PREFIX + r["text"], max_length=self.max_src, truncation=True)
        lab = self.tok(text_target=r["target"], max_length=self.max_tgt, truncation=True)
        enc["labels"] = lab["input_ids"]
        return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/flan-t5-base")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models/ca-extractor")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lora", action="store_true", help="train LoRA adapters instead of all weights")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
                         lora_dropout=0.05, target_modules=["q", "v"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    lr = args.lr or (1e-3 if args.lora else 3e-4)
    data = Path(args.data)
    train_ds = NoticeDataset(data / "train.jsonl", tok)
    val_ds = NoticeDataset(data / "val.jsonl", tok)

    out = Path(args.out)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model))
    trainer.train()

    final = trainer.model.merge_and_unload() if args.lora else trainer.model
    out.mkdir(parents=True, exist_ok=True)
    final.config.tie_word_embeddings = False   # Flan-T5 keeps a separate lm_head; silences a loading warning
    final.save_pretrained(out)
    tok.save_pretrained(out)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/train_log.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Saved fine-tuned model to {out}  (training log: reports/train_log.json)")


if __name__ == "__main__":
    main()
