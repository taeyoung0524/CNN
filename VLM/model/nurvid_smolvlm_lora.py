"""
NurViD 이미지 캡션 기반 SmolVLM LoRA Fine-tuning 파이프라인

실행 순서:
1) python nurvid_smolvlm_lora.py --mode make_template
2) dataset/nurvid_captions.jsonl에서 caption 직접 수정
3) python nurvid_smolvlm_lora.py --mode baseline
4) python nurvid_smolvlm_lora.py --mode train
5) python nurvid_smolvlm_lora.py --mode after
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel


IMAGE_DIR = "./dataset/selected_images"
TRAIN_JSONL = "./dataset/nurvid_captions.jsonl"
BASELINE_RESULT_CSV = "./results/baseline_smolvlm_results.csv"
LORA_OUTPUT_DIR = "./outputs/smolvlm_nurvid_lora"
AFTER_RESULT_CSV = "./results/after_lora_results.csv"

MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
PROMPT = "Describe this medical scene in detail."


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def build_messages_for_inference():
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def build_messages_for_training(caption):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": caption},
            ],
        },
    ]


def make_caption_template():
    ensure_dir(os.path.dirname(TRAIN_JSONL))

    image_files = [
        file for file in sorted(os.listdir(IMAGE_DIR))
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print(f"[WARN] 이미지가 없습니다: {IMAGE_DIR}")
        return

    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for file in image_files:
            sample = {
                "image": file,
                "caption": "A nurse performing a medical action in a hospital setting."
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[OK] Caption template saved to: {TRAIN_JSONL}")
    print("[NEXT] caption 값을 이미지 내용에 맞게 직접 수정하세요.")


class NurvidCaptionDataset(Dataset):
    def __init__(self, jsonl_path, image_dir):
        self.image_dir = image_dir
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = os.path.join(self.image_dir, item["image"])

        return {
            "image": load_image(image_path),
            "caption": item["caption"],
            "image_name": item["image"],
        }


@dataclass
class SmolVLMCollator:
    processor: AutoProcessor

    def __call__(self, examples):
        images = []
        texts = []

        for ex in examples:
            images.append(ex["image"])

            messages = build_messages_for_training(ex["caption"])

            text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
            )
            texts.append(text)

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch


def load_base_model_and_processor():
    device = get_device()
    dtype = get_dtype()

    print(f"[INFO] Loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"[INFO] Loading model: {MODEL_ID}")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if device == "cpu":
        model.to(device)

    return model, processor


def generate_caption(model, processor, image_path, max_new_tokens=80):
    image = load_image(image_path)

    messages = build_messages_for_inference()

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_len:]

    caption = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return caption.strip()


def run_inference(result_csv, use_lora=False):
    ensure_dir(os.path.dirname(result_csv))

    model, processor = load_base_model_and_processor()

    if use_lora:
        print(f"[INFO] Loading LoRA adapter from: {LORA_OUTPUT_DIR}")
        model = PeftModel.from_pretrained(model, LORA_OUTPUT_DIR)
        model.eval()

    image_files = [
        file for file in sorted(os.listdir(IMAGE_DIR))
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    rows = []

    for file in tqdm(image_files, desc="Generating captions"):
        image_path = os.path.join(IMAGE_DIR, file)
        caption = generate_caption(model, processor, image_path)

        rows.append({
            "image": file,
            "generated_caption": caption,
        })

        print(f"\n[{file}] {caption}")

    pd.DataFrame(rows).to_csv(
        result_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"[OK] Results saved to: {result_csv}")


def train_lora():
    ensure_dir(LORA_OUTPUT_DIR)

    if not os.path.exists(TRAIN_JSONL):
        raise FileNotFoundError(
            f"{TRAIN_JSONL} 파일이 없습니다. 먼저 --mode make_template 실행 후 caption을 수정하세요."
        )

    model, processor = load_base_model_and_processor()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = NurvidCaptionDataset(TRAIN_JSONL, IMAGE_DIR)
    collator = SmolVLMCollator(processor=processor)

    training_args = TrainingArguments(
        output_dir=LORA_OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

    model.save_pretrained(LORA_OUTPUT_DIR)
    processor.save_pretrained(LORA_OUTPUT_DIR)

    print(f"[OK] LoRA adapter saved to: {LORA_OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["make_template", "baseline", "train", "after"],
    )

    args = parser.parse_args()

    if args.mode == "make_template":
        make_caption_template()

    elif args.mode == "baseline":
        run_inference(BASELINE_RESULT_CSV, use_lora=False)

    elif args.mode == "train":
        train_lora()

    elif args.mode == "after":
        run_inference(AFTER_RESULT_CSV, use_lora=True)


if __name__ == "__main__":
    main()