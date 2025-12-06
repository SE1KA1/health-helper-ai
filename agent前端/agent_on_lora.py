import torch
import os
import re
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_PATH = "Qwen2.5-0.5B-Instruct"
LORA_MODEL_PATH = "health_advice_qwen0.5b_cpu_final"
DEVICE = "cpu"

def load_model_and_tokenizer():
    print("📌 正在加载模型，请稍候...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.float32,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    base_model.eval()
    finetuned_model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL_PATH,
        local_files_only=True
    )

    print("✅ 模型加载完成！")
    return finetuned_model, tokenizer

def generate_health_advice(model, tokenizer, habit, max_new_tokens=300):
    prompt = f"""用户有不良生活习惯：{habit}，请严格按照要求生成养生建议：
1. 仅生成3条建议，从1到3编号，格式为“数字、建议内容。”；
2. 每条建议简洁实用，10-20字，以句号结尾，无多余解释；
3. 内容紧扣{habit}，不重复、不截断，无任何寒暄或额外说明。"""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512  
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=1.5,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=4,
            use_cache=True,
            early_stopping=True,
            max_time=40.0
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    advice = response.split(prompt)[-1].strip()
    advice_sentences = re.findall(r'[123][、.：: ]*(.*?。)', advice)
    valid_sentences = []
    for sent in advice_sentences:
        sent = sent.strip()
        if sent and len(sent) >= 8 and sent not in valid_sentences:
            valid_sentences.append(sent)
    # 🔥 保留通用兜底建议（已移除熬夜专属部分）
    default_advice = [
        "规律作息，每天保证8小时睡眠。",
        "饮食均衡，多吃新鲜蔬菜水果。",
        "每天适度运动，增强身体抵抗力。"
        ]
    final_sentences = valid_sentences[:3]
    need_add = 3 - len(final_sentences)
    if need_add > 0:
        for adv in default_advice:
            if adv not in final_sentences and need_add > 0:
                final_sentences.append(adv)
                need_add -= 1
    final_advice = []
    for i, sent in enumerate(final_sentences[:3], 1):
        final_advice.append(f"{i}、{sent}")

    return "\n".join(final_advice)

def main():
    model, tokenizer = load_model_and_tokenizer()
    print("\n" + "="*80)
    print("🎯 养生建议助手 - 基于Qwen2.5-0.5B LoRA微调模型")
    print("💡 输入你的不良生活习惯，我会为你生成3条定制化养生建议（输入'退出'/'q'结束程序）")
    print("="*80 + "\n")
    while True:
        user_input = input("请输入你的不良生活习惯：").strip()
        if user_input.lower() in ["退出", "q", "quit", "exit"]:
            print("\n👋 感谢使用，祝你身体健康！")
            break
        if not user_input:
            print("🤖请跟我说下你的小问题吧\n")
            continue
        print("\n🤖 正在为你生成养生建议...\n")
        try:
            advice = generate_health_advice(model, tokenizer, user_input)
            print("✅ 养生建议：")
            print("-"*60)
            print(advice)
            print("-"*60 + "\n")
        except Exception as e:
            print(f"❌ 生成失败：{str(e)}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序已终止，感谢使用！")
    except Exception as e:
        print(f"\n❌ 程序异常：{str(e)}")
        print("💡 请检查模型路径是否正确，或重新运行程序。")