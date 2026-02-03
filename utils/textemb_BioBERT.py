import os, json, hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,L,1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B,H]
    denom = mask.sum(dim=1).clamp(min=1e-6)                         # [B,1]
    return summed / denom

def image_stem(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]

def hash_report(text: str) -> str:
    # Normalize lightly so identical reports hash the same
    t = (text or "").strip().lower()
    t = " ".join(t.split())  # collapse whitespace
    return hashlib.sha1(t.encode("utf-8")).hexdigest()

@torch.no_grad()
def precompute_unique_report_embeddings(
    dataloader,
    save_directory: str,
    model_name: str = "dmis-lab/biobert-base-cased-v1.1",
    max_length: int = 256,
    pooling: str = "mean",   # "mean" or "cls"
    device: str = "cuda",
    overwrite: bool = False,
    use_safetensors: bool = True,
):
    os.makedirs(save_directory, exist_ok=True)
    emb_path = os.path.join(save_directory, "reports_emb.npy")
    meta_path = os.path.join(save_directory, "reports_meta.json")
    map_path  = os.path.join(save_directory, "image_to_report_idx.json")

    if (not overwrite) and os.path.exists(emb_path) and os.path.exists(meta_path) and os.path.exists(map_path):
        print("[INFO] Found existing files; set overwrite=True to regenerate.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModel.from_pretrained(model_name, use_safetensors=use_safetensors)
    except ValueError as exc:
        raise ValueError(
            "Failed to load model weights. If you are on torch<2.6, set "
            "use_safetensors=True or upgrade torch to >=2.6."
        ) from exc
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    report_hash_to_idx = {}
    unique_reports = []          # list[str] in idx order
    image_to_idx = {}            # image_stem -> report_idx

    # 1) First pass: collect unique reports + image mapping
    for batch in dataloader:
        if isinstance(batch, list):
            texts = [b["text"] for b in batch]
            paths = [b["image_path"] for b in batch]
        else:
            texts = list(batch["text"]) if isinstance(batch["text"], (list, tuple)) else [batch["text"]]
            paths = list(batch["image_path"]) if isinstance(batch["image_path"], (list, tuple)) else [batch["image_path"]]

        for txt, pth in zip(texts, paths):
            h = hash_report(txt)
            if h not in report_hash_to_idx:
                report_hash_to_idx[h] = len(unique_reports)
                unique_reports.append(txt if txt is not None else "")
            image_to_idx[image_stem(str(pth))] = report_hash_to_idx[h]

    print(f"[INFO] Unique reports: {len(unique_reports)} (from {len(image_to_idx)} images)")

    # 2) Second pass: encode unique reports in batches
    all_emb = []
    bs = 32  # encoding batch size; you can increase if GPU allows
    for i in range(0, len(unique_reports), bs):
        chunk = unique_reports[i:i+bs]
        tok = tokenizer(
            chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        out = model(**tok)
        last_hidden = out.last_hidden_state  # [b,L,768]

        if pooling == "cls":
            pooled = last_hidden[:, 0, :]
        elif pooling == "mean":
            pooled = mean_pool(last_hidden, tok["attention_mask"])
        else:
            raise ValueError("pooling must be 'mean' or 'cls'")

        all_emb.append(pooled.detach().cpu().numpy().astype(np.float32))

    reports_emb = np.concatenate(all_emb, axis=0)  # [N_unique, 768]
    np.save(emb_path, reports_emb)

    # Save metadata + mappings
    meta = {
        "model_name": model_name,
        "max_length": max_length,
        "pooling": pooling,
        "embedding_dim": int(reports_emb.shape[1]),
        "num_unique_reports": int(reports_emb.shape[0]),
        "report_hash_to_idx": report_hash_to_idx,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    with open(map_path, "w") as f:
        json.dump(image_to_idx, f)

    print(f"[DONE] Saved:\n- {emb_path}\n- {meta_path}\n- {map_path}")
