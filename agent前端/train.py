import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm
import re
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType

LOCAL_MODEL_PATH = os.path.abspath("C:/Users/wyl/Desktop/ICT/VScodeICTprojects/modelscope_cache/qwen2.5-7b-instruct/qwen/Qwen2___5-7B-Instruct")
print(f"📌 本地模型路径：{LOCAL_MODEL_PATH}")
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(f"本地模型路径不存在：{LOCAL_MODEL_PATH}")

# 设置设备（CPU专用配置）
device = "cpu"
torch.set_default_device(device)
torch.backends.cudnn.enabled = False

CONFIG = {
    "lora_rank": 2,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "max_seq_len": 256,
    "batch_size": 1,
    "lr": 2e-4,
    "epochs": 5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "early_stop_patience": 2,
    "early_stop_min_delta": 1e-5,
    "data_path": "./health_data.txt",
    "test_size": 10,
    "val_size": 10,
    "output_dir": "./qwen_lora_finetune_cpu",
    "ckpt_name": "health_advice_qwen_lora_cpu",
}

class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-5, restore_best_weights=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        self.best_model_path = os.path.join(CONFIG["output_dir"], "tmp_best_lora_model")

    def __call__(self, current_loss, model, epoch):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                model.save_pretrained(self.best_model_path)
            if self.verbose:
                print(f"✅ 验证损失改进 ({self.best_loss:.6f})，保存最佳LoRA权重")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️  验证损失无改进，计数器: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n🛑 触发早停！最佳轮数：{self.best_epoch+1}，最佳损失：{self.best_loss:.6f}")
        return self.early_stop

    def restore_best_model(self, model):
        if self.restore_best_weights and os.path.exists(self.best_model_path):
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, self.best_model_path, local_files_only=True)
                if self.verbose:
                    print(f"\n✅ 恢复最佳LoRA模型（第{self.best_epoch+1}轮）")
                import shutil
                shutil.rmtree(self.best_model_path)
            except Exception as e:
                print(f"\n❌ 恢复最佳模型失败：{e}")
        return model

def clean_chinese_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\u4e00-\u9fff0-9，。！？；：""''（）【】《》、·…—]', '', text)
    return text[:200]

def load_data_from_txt(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|||")
            if len(parts) != 2:
                print(f"警告：第{line_num}行格式错误 -> {line}")
                continue
            bad_habit = clean_chinese_text(parts[0].strip())
            advice = clean_chinese_text(parts[1].strip())
            if bad_habit and advice:
                prompt = [
                    {"role": "user", "content": f"用户有不良生活习惯：{bad_habit}，请给出养生建议。"},
                    {"role": "assistant", "content": advice}
                ]
                data.append(prompt)
    print(f"✅ 加载{len(data)}条养生数据")
    return data

def build_dataset():
    raw_data = load_data_from_txt(CONFIG["data_path"])
    random.shuffle(raw_data)
    
    test_data = raw_data[:CONFIG["test_size"]]
    val_data = raw_data[CONFIG["test_size"]:CONFIG["test_size"]+CONFIG["val_size"]]
    train_data = raw_data[CONFIG["test_size"]+CONFIG["val_size"]:]
    
    print(f"\n数据拆分：训练集{len(train_data)} | 验证集{len(val_data)} | 测试集{len(test_data)}")
    return train_data, val_data, test_data

def format_prompt(data, tokenizer):
    """格式化Qwen的对话输入"""
    formatted_texts = []
    for item in data:
        formatted = tokenizer.apply_chat_template(
            item,
            tokenize=False,
            add_generation_prompt=False
        )
        encoding = tokenizer(
            formatted,
            truncation=True,
            max_length=CONFIG["max_seq_len"],
            padding="max_length",
            return_tensors="pt"
        )
        formatted_texts.append({
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        })
    return formatted_texts

def init_qwen_model():
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=True,
        cache_dir=None
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        cache_dir=None,
        revision=None,
        mirror=None
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"],
        bias="none",
        inference_mode=False,
    )
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model, tokenizer

def train_lora_model():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    train_data, val_data, test_data = build_dataset()
    model, tokenizer = init_qwen_model()
    train_formatted = format_prompt(train_data, tokenizer)
    val_formatted = format_prompt(val_data, tokenizer)
    test_formatted = format_prompt(test_data, tokenizer)
    class HealthDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return {
                "input_ids": self.data[idx]["input_ids"],
                "attention_mask": self.data[idx]["attention_mask"],
                "labels": self.data[idx]["labels"]
            }
    
    train_loader = DataLoader(
        HealthDataset(train_formatted), 
        batch_size=CONFIG["batch_size"], 
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        HealthDataset(val_formatted), 
        batch_size=CONFIG["batch_size"],
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        HealthDataset(test_formatted), 
        batch_size=CONFIG["batch_size"],
        num_workers=0,
        pin_memory=False
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        eps=1e-8
    )
    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    early_stopper = EarlyStopping(
        patience=CONFIG["early_stop_patience"],
        min_delta=CONFIG["early_stop_min_delta"]
    )
    
    best_val_loss = float('inf')
    model.train()
    
    for epoch in range(CONFIG["epochs"]):
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch in pbar:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                total_val_loss += outputs.loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                total_test_loss += outputs.loss.item()
        avg_test_loss = total_test_loss / len(test_loader)
        
        if early_stopper(avg_val_loss, model, epoch):
            break
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(os.path.join(CONFIG["output_dir"], f"{CONFIG['ckpt_name']}_best"))
        
        print(f"\nEpoch [{epoch+1}/{CONFIG['epochs']}]")
        print(f"  训练损失: {avg_train_loss:.6f}")
        print(f"  验证损失: {avg_val_loss:.6f}")
        print(f"  测试损失: {avg_test_loss:.6f}")
        model.train()
    
    model = early_stopper.restore_best_model(model)
    final_model_path = os.path.join(CONFIG["output_dir"], f"{CONFIG['ckpt_name']}_final")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n✅ Qwen-LoRA微调完成！模型保存至：{final_model_path}")
    return model, tokenizer

def generate_health_advice(model, tokenizer, bad_habit, max_new_tokens=50):
    """Qwen推理函数（CPU适配）"""
    chat = [{"role": "user", "content": f"用户有不良生活习惯：{bad_habit}，请给出养生建议。"}]
    prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CONFIG["max_seq_len"] - max_new_tokens
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=1,
            length_penalty=1.0,
            use_cache=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    advice = response.split("assistant\n")[-1].strip()
    return advice

if __name__ == "__main__":
    print("=" * 80)
    print("         Qwen2.5-7B-Instruct（本地模型+纯CPU）LoRA 养生建议微调")
    print("=" * 80)
    
    lora_model, tokenizer = train_lora_model()
    
    test_cases = [
        "每天熬夜到凌晨1点，早上7点起床",
        "久坐办公室，几乎不运动",
        "经常吃辛辣食物，容易上火"
    ]
    
    print("\n===== 养生建议生成测试 =====")
    for case in test_cases:
        advice = generate_health_advice(lora_model, tokenizer, case)
        print(f"\n❌ 不良习惯：{case}")
        print(f"✅ 养生建议：{advice}")
        print("-" * 50)
    
    print("\n===== 加载微调模型推理 =====")
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        local_files_only=True
    )
    finetuned_model = PeftModel.from_pretrained(
        base_model,
        os.path.join(CONFIG["output_dir"], f"{CONFIG['ckpt_name']}_final"),
        local_files_only=True
    )
    infer_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(CONFIG["output_dir"], f"{CONFIG['ckpt_name']}_final"),
        trust_remote_code=True,
        local_files_only=True
    )
    test_advice = generate_health_advice(
        finetuned_model,
        infer_tokenizer,
        "长期不吃早餐，饮食不规律"
    )
    print(f"\n❌ 不良习惯：长期不吃早餐，饮食不规律")
    print(f"✅ 养生建议：{test_advice}")