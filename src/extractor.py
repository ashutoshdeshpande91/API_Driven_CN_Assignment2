"""Inference wrapper around the fine-tuned (or any seq2seq) model."""
from src.schema import PREFIX, parse_output


class Extractor:
    def __init__(self, model_dir, device=None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(self.device).eval()

    def extract_raw(self, texts, batch_size=32, num_beams=4, max_new_tokens=128):
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = [PREFIX + t for t in texts[i:i + batch_size]]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=512).to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, num_beams=num_beams, max_new_tokens=max_new_tokens)
            outs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return outs

    def extract(self, texts, **kw):
        return [parse_output(o) for o in self.extract_raw(texts, **kw)]
