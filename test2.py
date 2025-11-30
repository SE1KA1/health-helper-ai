import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.processors import TemplateProcessing
from mindspore.dataset import GeneratorDataset
from mindspore.common.initializer import Normal, initializer
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
import random
import os
import json
import re

# 设置MindSpore上下文
ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_device("CPU")

# ======================== 替换后的分词器核心配置与工具函数 ========================
# 定义特殊token（全局固定ID，避免动态获取不一致）
SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[SOS]": 2,
    "[EOS]": 3
}

# 中文文本清洗函数（适配中文，过滤乱码/特殊字符）
def clean_chinese_text(text):
    """清洗中文文本，保留中文、数字、常用标点"""
    text = re.sub(r'\s+', ' ', text).strip()  # 移除多余空格
    text = re.sub(r'[^\u4e00-\u9fff0-9，。！？；：""''（）【】《》、·…—]', '', text)  # 保留核心字符
    return text[:100]  # 限制最大长度，避免编码异常

# 新增：分词结果可视化函数
def visualize_tokenization(tokenizer, text, desc="分词结果"):
    """可视化文本的分词结果（打印token、ID、详细信息）"""
    print(f"\n===== {desc} =====")
    print(f"原始文本：{text}")
    encoded = tokenizer.encode(text)
    # 打印核心分词信息
    print(f"Token列表：{encoded.tokens}")
    print(f"ID列表：{encoded.ids}")
    print(f"分词长度：{len(encoded.ids)}")
    print(f"特殊token占比：{sum(1 for id in encoded.ids if id in SPECIAL_TOKENS.values())}/{len(encoded.ids)}")
    # 打印子词映射（中文BPE子词）
    subword_info = []
    for token, id in zip(encoded.tokens, encoded.ids):
        if token.startswith("##"):
            subword_info.append(f"{token} (ID:{id}, 子词)")
        else:
            subword_info.append(f"{token} (ID:{id}, 基础词)")
    print(f"子词详情：{' | '.join(subword_info)}")
    print("-" * 50)

# 训练中文优化版BPE分词器（替换原分词器训练逻辑，新增分词可视化）
def train_health_tokenizer(texts, vocab_size):
    """训练适配中文的BPE分词器（中文友好版）"""
    print(f"\n开始训练中文BPE分词器（词汇表大小：{vocab_size}）...")
    
    # 初始化分词器（指定unk_token）
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # 文本归一化（统一全角/半角、字符编码）
    tokenizer.normalizer = Sequence([NFKC()])
    
    # 中文友好的预分词策略（标点+空格切分，替代ByteLevel）
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Punctuation(),  # 标点符号独立切分
        pre_tokenizers.Whitespace()    # 空格切分
    ])
    
    # 训练配置（指定特殊token，适配中文子词）
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS.keys()),
        min_frequency=1,
        show_progress=True,
        continuing_subword_prefix="##",  # 中文子词前缀
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()  # 兼容中文编码
    )
    
    # 文本编码保护（避免中文乱码）
    cleaned_texts = [text.encode("utf-8").decode("utf-8") for text in texts]
    tokenizer.train_from_iterator(cleaned_texts, trainer=trainer)
    
    # 自动添加[SOS]/[EOS]的后处理逻辑
    tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        pair="[SOS] $A [EOS] $B [EOS]",
        special_tokens=[
            ("[SOS]", SPECIAL_TOKENS["[SOS]"]),
            ("[EOS]", SPECIAL_TOKENS["[EOS]"]),
        ],
    )
    
    # 强制确认特殊token ID映射
    for token, token_id in SPECIAL_TOKENS.items():
        tokenizer.add_special_tokens([token])
        tokenizer.token_to_id(token)
    
    # 保存分词器
    tokenizer.save("health_tokenizer.json")
    print(f"中文分词器训练完成，保存路径：health_tokenizer.json")
    print(f"特殊token ID映射：{SPECIAL_TOKENS}")
    
    # 新增：测试分词器并可视化结果（取前3条文本测试）
    test_texts = cleaned_texts[:3] if len(cleaned_texts) >=3 else cleaned_texts
    for i, test_text in enumerate(test_texts):
        visualize_tokenization(tokenizer, test_text, desc=f"训练后分词测试 - 示例{i+1}")
    
    return tokenizer

# ======================== 原程序核心配置 ========================
CONFIG = {
    "vocab_size": 8000,
    "max_input_len": 64,
    "max_output_len": 128,
    "d_model": 128,
    "num_heads": 4,
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "d_ff": 256,
    "dropout_rate": 0.1,
    "batch_size": 2,
    "lr": 5e-5,
    "epochs": 200,
    "warmup_steps": 20,
    "temperature": 0.9,
    "clip_norm": 1.0,
    "dtype": "int32",
    "export_format": "CKPT",
    "export_dir": "./exported_model",
    "export_filename": "health_advice_transformer",
    "data_path": "./health_data.txt"
}

# 类型映射字典
DTYPE_MAP = {
    "int32": ms.int32,
    "float32": ms.float32,
    "bool": ms.bool_
}

# ======================== 数据加载与数据集类（新增分词可视化） ========================
def load_data_from_txt(file_path):
    """读取TXT数据（集成中文清洗）"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|||")
            if len(parts) != 2:
                print(f"警告：第{line_num}行格式错误，已跳过 -> {line}")
                continue
            # 清洗中文文本
            bad_habit = clean_chinese_text(parts[0].strip())
            advice = clean_chinese_text(parts[1].strip())
            if bad_habit and advice:
                data.append((bad_habit, advice))
    print(f"成功从{file_path}读取{len(data)}条健康数据（已清洗中文）")
    
    # 新增：打印前3条原始数据的清洗结果
    print("\n===== 数据清洗示例 =====")
    for i, (habit, advice) in enumerate(data[:3]):
        print(f"示例{i+1} - 不良习惯：{habit}")
        print(f"示例{i+1} - 养生建议：{advice}")
    print("-" * 50)
    
    return data

def build_health_dataset():
    """构建训练/验证/测试数据集"""
    raw_data = load_data_from_txt(CONFIG["data_path"])
    random.shuffle(raw_data)
    
    # 拆分数据集
    test_size = 30
    val_size = 30
    test_data = raw_data[:test_size]
    val_data = raw_data[test_size:test_size+val_size]
    train_data = raw_data[test_size+val_size:]
    
    print("\n数据拆分结果：")
    print(f"训练集（{len(train_data)}条）：{[item[0] for item in train_data[:5]]}...")
    print(f"验证集（{len(val_data)}条）：{[item[0] for item in val_data[:3]]}...")
    print(f"测试集（{len(test_data)}条）：{[item[0] for item in test_data[:3]]}...")
    
    return train_data, val_data, test_data

class HealthDataset:
    """数据集类（适配新分词器，新增分词结果打印）"""
    def __init__(self, data, tokenizer, max_input_len, max_output_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        # 使用固定的特殊token ID
        self.sos_id = SPECIAL_TOKENS["[SOS]"]
        self.eos_id = SPECIAL_TOKENS["[EOS]"]
        self.pad_id = SPECIAL_TOKENS["[PAD]"]
        self.unk_id = SPECIAL_TOKENS["[UNK]"]
        # 标记是否打印过分词示例（避免重复打印）
        self._print_token_example = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        
        # 输入文本编码（过滤重复特殊token）
        src_encoded = self.tokenizer.encode(src_text)
        src_ids_clean = [id for id in src_encoded.ids if id not in [self.sos_id, self.eos_id, self.pad_id, self.unk_id]]
        src_ids = [self.sos_id] + src_ids_clean[:self.max_input_len-2] + [self.eos_id]
        src_pad_len = self.max_input_len - len(src_ids)
        src_ids += [self.pad_id] * src_pad_len
        src_mask = [1]*(len(src_ids)-src_pad_len) + [0]*src_pad_len
        
        # 输出文本编码
        tgt_encoded = self.tokenizer.encode(tgt_text)
        tgt_ids_clean = [id for id in tgt_encoded.ids if id not in [self.sos_id, self.eos_id, self.pad_id, self.unk_id]]
        tgt_input_ids = [self.sos_id] + tgt_ids_clean[:self.max_output_len-2]
        tgt_input_pad_len = self.max_output_len - len(tgt_input_ids)
        tgt_input_ids += [self.pad_id] * tgt_input_pad_len
        
        tgt_target_ids = tgt_ids_clean[:self.max_output_len-2] + [self.eos_id]
        tgt_target_pad_len = self.max_output_len - len(tgt_target_ids)
        tgt_target_ids += [self.pad_id] * tgt_target_pad_len
        
        tgt_mask = [1]*(len(tgt_input_ids)-tgt_input_pad_len) + [0]*tgt_input_pad_len
        
        # 新增：打印第一条数据的完整分词编码结果（仅打印一次）
        if self._print_token_example and idx == 0:
            print("\n===== 数据集编码示例（第一条数据） =====")
            print(f"原始输入文本：{src_text}")
            print(f"输入ID（含SOS/EOS/PAD）：{src_ids}")
            print(f"输入MASK：{src_mask}")
            print(f"原始输出文本：{tgt_text}")
            print(f"输出输入ID（含SOS/PAD）：{tgt_input_ids}")
            print(f"输出目标ID（含EOS/PAD）：{tgt_target_ids}")
            print(f"输出MASK：{tgt_mask}")
            print("-" * 50)
            self._print_token_example = False
        
        return (
            np.array(src_ids, np.int32),
            np.array(src_mask, np.int32),
            np.array(tgt_input_ids, np.int32),
            np.array(tgt_mask, np.int32),
            np.array(tgt_target_ids, np.int32)
        )

# ======================== 模型核心组件（完整保留原逻辑） ========================
class MultiHeadAttention(nn.Cell):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Dense(d_model, d_model)
        self.w_k = nn.Dense(d_model, d_model)
        self.w_v = nn.Dense(d_model, d_model)
        self.w_o = nn.Dense(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(axis=-1)
        self.sqrt_dk = ms.Tensor(np.sqrt(self.d_k), ms.float32)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.w_q, self.w_k, self.w_v, self.w_o]:
            layer.weight.set_data(initializer(Normal(0.02), layer.weight.shape))
            if layer.bias is not None:
                layer.bias.set_data(initializer('zeros', layer.bias.shape))

    def construct(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        len_q = q.shape[1]
        len_k = k.shape[1]
        len_v = v.shape[1]
        
        q = self.w_q(q).reshape(batch_size, len_q, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        k = self.w_k(k).reshape(batch_size, len_k, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        v = self.w_v(v).reshape(batch_size, len_v, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        
        scores = ops.matmul(q, k.transpose(0, 1, 3, 2)) / self.sqrt_dk
        
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == False, -1e9)
        
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        
        output = ops.matmul(attn, v)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, len_q, -1)
        output = self.w_o(output)
        
        return output

class FeedForward(nn.Cell):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Dense(d_model, d_ff)
        self.fc2 = nn.Dense(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        
        self._initialize_weights()

    def _initialize_weights(self):
        self.fc1.weight.set_data(initializer(Normal(0.02), self.fc1.weight.shape))
        self.fc2.weight.set_data(initializer(Normal(0.02), self.fc2.weight.shape))
        if self.fc1.bias is not None:
            self.fc1.bias.set_data(initializer('zeros', self.fc1.bias.shape))
        if self.fc2.bias is not None:
            self.fc2.bias.set_data(initializer('zeros', self.fc2.bias.shape))

    def construct(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class EncoderLayer(nn.Cell):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm((d_model,), epsilon=1e-6)
        self.norm2 = nn.LayerNorm((d_model,), epsilon=1e-6)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def construct(self, x, mask):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class Encoder(nn.Cell):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.CellList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm((d_model,), epsilon=1e-6)

    def construct(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Cell):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm((d_model,), epsilon=1e-6)
        self.norm2 = nn.LayerNorm((d_model,), epsilon=1e-6)
        self.norm3 = nn.LayerNorm((d_model,), epsilon=1e-6)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def construct(self, x, enc_out, tgt_mask, src_mask):
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn1))
        
        attn2 = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout2(attn2))
        
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x

class Decoder(nn.Cell):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.CellList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm((d_model,), epsilon=1e-6)

    def construct(self, x, enc_out, tgt_mask, src_mask):
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.norm(x)

class PositionalEncoding(nn.Cell):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = ms.Tensor(pe[np.newaxis, ...], ms.float32)

    def construct(self, x):
        return x + self.pe[:, :x.shape[1], :]

class TransformerModel(nn.Cell):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, 
                 num_encoder_layers, num_decoder_layers, 
                 pad_id, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder = Encoder(d_model, num_heads, d_ff, num_encoder_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_decoder_layers, dropout)
        self.fc_out = nn.Dense(d_model, vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.sqrt_dmodel = ms.Tensor(np.sqrt(d_model), ms.float32)
        
        self._initialize_weights()

    def _initialize_weights(self):
        self.embedding.embedding_table.set_data(initializer(Normal(0.02), self.embedding.embedding_table.shape))
        self.fc_out.weight.set_data(initializer(Normal(0.02), self.fc_out.weight.shape))
        if self.fc_out.bias is not None:
            self.fc_out.bias.set_data(initializer('zeros', self.fc_out.bias.shape))

    def _generate_subsequent_mask(self, seq_len):
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(np.bool_)
        return ms.Tensor(mask, ms.bool_)

    def construct(self, src_ids, src_mask, tgt_ids, tgt_mask):
        batch_size = src_ids.shape[0]
        src_len = src_ids.shape[1]
        tgt_len = tgt_ids.shape[1]
        
        src_emb = self.embedding(src_ids) * self.sqrt_dmodel
        src_emb = self.dropout(self.pos_encoding(src_emb))
        enc_out = self.encoder(src_emb, src_mask)
        
        tgt_emb = self.embedding(tgt_ids) * self.sqrt_dmodel
        tgt_emb = self.dropout(self.pos_encoding(tgt_emb))
        
        subsequent_mask = self._generate_subsequent_mask(tgt_len)
        tgt_pad_mask = tgt_mask.unsqueeze(1)
        tgt_pad_mask = ops.tile(tgt_pad_mask, (1, tgt_len, 1))
        tgt_mask_final = ops.logical_and(tgt_pad_mask, ~subsequent_mask)
        
        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask_final, src_mask)
        output = self.fc_out(dec_out)
        
        return output

# ======================== 训练相关工具函数 ========================
class FixedLR(LearningRateSchedule):
    def __init__(self, lr):
        super().__init__()
        self.lr = ms.Tensor(lr, ms.float32)

    def construct(self, step):
        return self.lr

def calculate_loss(model, dataset, loss_fn, pad_id):
    """计算数据集平均损失"""
    model.set_train(False)
    total_loss = 0.0
    step = 0
    
    for batch in dataset.create_tuple_iterator():
        src_ids, src_mask, tgt_input_ids, tgt_mask, tgt_target_ids = batch
        
        src_mask = src_mask.astype(ms.bool_)
        tgt_mask = tgt_mask.astype(ms.bool_)
        
        logits = model(src_ids, src_mask, tgt_input_ids, tgt_mask)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        tgt_target_flat = tgt_target_ids.reshape(-1)
        
        loss = loss_fn(logits_flat, tgt_target_flat)
        loss_np = loss.asnumpy()
        
        if not np.isnan(loss_np) and not np.isinf(loss_np):
            total_loss += loss_np
        step += 1
    
    avg_loss = total_loss / step if step > 0 else 0.0
    model.set_train(True)
    return avg_loss

def export_model(model, config, pad_id):
    """导出模型（适配CPU，优先CKPT格式）"""
    os.makedirs(config["export_dir"], exist_ok=True)
    model.set_train(False)
    
    actual_dtype = DTYPE_MAP[config["dtype"]]
    
    try:
        if config["export_format"].upper() in ["MINDIR", "AIR"]:
            batch_size = 1
            src_ids = ms.Tensor(np.ones((batch_size, config["max_input_len"]), dtype=np.int32) * pad_id, actual_dtype)
            src_mask = ms.Tensor(np.ones((batch_size, config["max_input_len"]), dtype=np.bool_), ms.bool_)
            tgt_ids = ms.Tensor(np.ones((batch_size, config["max_output_len"]), dtype=np.int32) * pad_id, actual_dtype)
            tgt_mask = ms.Tensor(np.ones((batch_size, config["max_output_len"]), dtype=np.bool_), ms.bool_)
            
            export_path = os.path.join(config["export_dir"], config["export_filename"])
            ms.export(
                model,
                src_ids,
                src_mask,
                tgt_ids,
                tgt_mask,
                file_name=export_path,
                file_format=config["export_format"]
            )
            print(f"\n模型导出完成！路径: {export_path}.{config['export_format'].lower()}")
            
        else:
            ckpt_path = os.path.join(config["export_dir"], f"{config['export_filename']}.ckpt")
            ms.save_checkpoint(model, ckpt_path)
            print(f"\n模型保存为CKPT格式: {ckpt_path}")
            
    except Exception as e:
        print(f"\n注意：{config['export_format']}格式导出失败，错误：{str(e)[:200]}")
        print("降级保存为CKPT格式...")
        ckpt_path = os.path.join(config["export_dir"], f"{config['export_filename']}_fallback.ckpt")
        ms.save_checkpoint(model, ckpt_path)
        print(f"CKPT模型已保存至: {ckpt_path}")
    
    print(f"输入说明:")
    print(f"  - src_ids: [batch_size, max_input_len] (int32)")
    print(f"  - src_mask: [batch_size, max_input_len] (bool)")
    print(f"  - tgt_ids: [batch_size, max_output_len] (int32)")
    print(f"  - tgt_mask: [batch_size, max_output_len] (bool)")
    print(f"输出说明:")
    print(f"  - logits: [batch_size, max_output_len, vocab_size] (float32)")

# ======================== 核心训练与推理逻辑 ========================
def train_model():
    """完整训练流程"""
    # 数据准备
    train_data, val_data, test_data = build_health_dataset()
    all_texts = [item[0] for item in train_data] + [item[1] for item in train_data]
    
    # 训练新分词器（替换原逻辑）
    tokenizer = train_health_tokenizer(all_texts, CONFIG["vocab_size"])
    pad_id = SPECIAL_TOKENS["[PAD]"]  # 使用固定PAD ID
    
    # 构建数据集
    train_ds = HealthDataset(train_data, tokenizer, CONFIG["max_input_len"], CONFIG["max_output_len"])
    val_ds = HealthDataset(val_data, tokenizer, CONFIG["max_input_len"], CONFIG["max_output_len"])
    test_ds = HealthDataset(test_data, tokenizer, CONFIG["max_input_len"], CONFIG["max_output_len"])
    
    ms_train_ds = GeneratorDataset(
        train_ds,
        column_names=["src_ids", "src_mask", "tgt_input_ids", "tgt_mask", "tgt_target_ids"],
        shuffle=True
    ).batch(CONFIG["batch_size"], drop_remainder=False)
    
    ms_val_ds = GeneratorDataset(
        val_ds,
        column_names=["src_ids", "src_mask", "tgt_input_ids", "tgt_mask", "tgt_target_ids"],
        shuffle=False
    ).batch(CONFIG["batch_size"], drop_remainder=False)
    
    ms_test_ds = GeneratorDataset(
        test_ds,
        column_names=["src_ids", "src_mask", "tgt_input_ids", "tgt_mask", "tgt_target_ids"],
        shuffle=False
    ).batch(CONFIG["batch_size"], drop_remainder=False)
    
    # 初始化模型
    model = TransformerModel(
        vocab_size=CONFIG["vocab_size"],
        d_model=CONFIG["d_model"],
        num_heads=CONFIG["num_heads"],
        d_ff=CONFIG["d_ff"],
        num_encoder_layers=CONFIG["num_encoder_layers"],
        num_decoder_layers=CONFIG["num_decoder_layers"],
        pad_id=pad_id,
        dropout=CONFIG["dropout_rate"]
    )
    
    # 优化器与损失函数
    lr_schedule = FixedLR(CONFIG["lr"])
    optimizer = nn.Adam(
        model.trainable_params(),
        learning_rate=lr_schedule,
        beta1=0.9,
        beta2=0.98,
        eps=1e-9,
        use_lazy=False,
        use_amsgrad=False,
        weight_decay=0.0
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    # 训练网络封装
    class TrainNet(nn.Cell):
        def __init__(self, network, loss_fn, optimizer):
            super().__init__()
            self.network = network
            self.loss_fn = loss_fn
            self.optimizer = optimizer
            self.grad_fn = ops.value_and_grad(self.forward_fn, None, optimizer.parameters)

        def forward_fn(self, src_ids, src_mask, tgt_input_ids, tgt_mask, tgt_target_ids):
            logits = self.network(src_ids, src_mask, tgt_input_ids, tgt_mask)
            logits_flat = logits.reshape(-1, CONFIG["vocab_size"])
            tgt_target_flat = tgt_target_ids.reshape(-1)
            loss = self.loss_fn(logits_flat, tgt_target_flat)
            return loss

        def construct(self, src_ids, src_mask, tgt_input_ids, tgt_mask, tgt_target_ids):
            loss, grads = self.grad_fn(src_ids, src_mask, tgt_input_ids, tgt_mask, tgt_target_ids)
            grads = tuple([ops.clip_by_norm(g, CONFIG["clip_norm"]) for g in grads])
            self.optimizer(grads)
            return loss
    
    # 开始训练
    train_net = TrainNet(model, loss_fn, optimizer)
    train_net.set_train(True)
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG["epochs"]):
        total_loss = 0.0
        step = 0
        pbar = tqdm(ms_train_ds.create_tuple_iterator(), desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch in pbar:
            src_ids, src_mask, tgt_input_ids, tgt_mask, tgt_target_ids = batch
            
            src_mask = src_mask.astype(ms.bool_)
            tgt_mask = tgt_mask.astype(ms.bool_)
            
            loss = train_net(src_ids, src_mask, tgt_input_ids, tgt_mask, tgt_target_ids)
            
            loss_np = loss.asnumpy()
            if np.isnan(loss_np) or np.isinf(loss_np):
                loss_np = 0.0
            total_loss += loss_np
            step += 1
            pbar.set_postfix({"loss": f"{loss_np:.4f}"})
        
        # 计算损失
        train_avg_loss = total_loss / step if step > 0 else 0.0
        val_avg_loss = calculate_loss(model, ms_val_ds, loss_fn, pad_id)
        test_avg_loss = calculate_loss(model, ms_test_ds, loss_fn, pad_id)
        
        # 保存最佳模型
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            ms.save_checkpoint(model, "health_advice_model_best.ckpt")
            print(f"  -> 最佳验证损失更新，已保存模型: health_advice_model_best.ckpt")
        
        # 打印日志
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}]")
        print(f"  训练集平均损失: {train_avg_loss:.4f}")
        print(f"  验证集平均损失: {val_avg_loss:.4f}")
        print(f"  测试集平均损失: {test_avg_loss:.4f}")
        print(f"  最佳验证损失: {best_val_loss:.4f}")
    
    # 保存最终模型
    ms.save_checkpoint(model, "health_advice_model_final.ckpt")
    print("\n最终模型检查点保存完成：health_advice_model_final.ckpt")
    
    # 导出模型
    export_model(model, CONFIG, pad_id)
    
    return model, tokenizer

def generate_advice(model, tokenizer, input_text, config):
    """生成养生建议（适配新分词器，新增生成过程分词可视化）"""
    model.set_train(False)
    sos_id = SPECIAL_TOKENS["[SOS]"]
    eos_id = SPECIAL_TOKENS["[EOS]"]
    pad_id = SPECIAL_TOKENS["[PAD]"]
    actual_dtype = DTYPE_MAP[config["dtype"]]
    
    # 清洗并编码输入文本
    input_text = clean_chinese_text(input_text)
    # 新增：可视化输入文本的分词过程
    visualize_tokenization(tokenizer, input_text, desc=f"生成建议 - 输入文本分词")
    
    src_encoded = tokenizer.encode(input_text)
    src_ids_clean = [id for id in src_encoded.ids if id not in [sos_id, eos_id, pad_id, SPECIAL_TOKENS["[UNK]"]]]
    src_ids = [sos_id] + src_ids_clean[:config["max_input_len"]-2] + [eos_id]
    src_pad_len = config["max_input_len"] - len(src_ids)
    src_ids += [pad_id] * src_pad_len
    src_mask = [True]*(len(src_ids)-src_pad_len) + [False]*src_pad_len
    
    src_ids = ms.Tensor([src_ids], actual_dtype)
    src_mask = ms.Tensor([src_mask], ms.bool_)
    
    # 初始化生成
    tgt_ids = ms.Tensor([[sos_id]], actual_dtype)
    generate_steps = []  # 记录每一步生成的token
    
    for step in range(config["max_output_len"]-1):
        if tgt_ids.shape[1] >= config["max_output_len"]:
            break
            
        tgt_mask = ms.Tensor([[True]*tgt_ids.shape[1]], ms.bool_)
        logits = model(src_ids, src_mask, tgt_ids, tgt_mask)
        
        next_token = ops.argmax(logits[:, -1, :], dim=-1)
        next_token = next_token.unsqueeze(1).astype(actual_dtype)
        generate_steps.append(next_token.asnumpy()[0,0])
        
        tgt_ids = ops.concat([tgt_ids, next_token], axis=1)
        
        if next_token.asnumpy()[0,0] == eos_id:
            break
    
    # 新增：打印生成过程的token ID序列
    print(f"\n===== 生成过程详情 =====")
    print(f"生成步骤数：{len(generate_steps)}")
    print(f"生成token ID序列：{generate_steps}")
    print(f"是否生成EOS：{eos_id in generate_steps}")
    print("-" * 50)
    
    # 解码结果（过滤特殊token）
    output_ids = tgt_ids.asnumpy()[0].tolist()
    output_ids = [id for id in output_ids if id not in [sos_id, eos_id, pad_id, SPECIAL_TOKENS["[UNK]"]]]
    advice = tokenizer.decode(output_ids)
    
    # 新增：可视化输出结果的分词
    visualize_tokenization(tokenizer, advice, desc=f"生成建议 - 输出文本分词")
    
    # 兜底逻辑（避免空输出）
    if not advice:
        advice = "建议规律作息，合理饮食，适度运动，保持良好生活习惯。"
    
    return advice

def load_model_for_inference(ckpt_path, tokenizer_path, config):
    """加载模型用于推理"""
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_id = SPECIAL_TOKENS["[PAD]"]
    
    model = TransformerModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
        pad_id=pad_id,
        dropout=config["dropout_rate"]
    )
    
    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)
    
    return model, tokenizer

# ======================== 主函数 ========================
if __name__ == "__main__":
    print("=" * 60)
    print("         养生建议生成模型（中文优化版 + 分词可视化）")
    print("=" * 60)
    
    # 训练模型
    model, tokenizer = train_model()
    
    # 测试生成（新增分词可视化）
    test_cases = [
        "每天熬夜到凌晨1点，早上7点起床",
        "久坐办公室，几乎不运动",
        "经常吃辛辣食物，容易上火"
    ]
    
    print("\n===== 养生建议生成测试（含分词可视化） =====")
    for case in test_cases:
        advice = generate_advice(model, tokenizer, case, CONFIG)
        print(f"\n【最终结果】")
        print(f"输入：{case}")
        print(f"建议：{advice}")
        print("=" * 50)
    
    # 保存配置文件
    config_path = os.path.join(CONFIG["export_dir"], "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=4)
    print(f"\n配置文件已保存至: {config_path}")
    
    # 测试加载模型
    print("\n===== 测试加载保存的模型 =====")
    try:
        loaded_model, loaded_tokenizer = load_model_for_inference(
            "health_advice_model_final.ckpt",
            "health_tokenizer.json",
            CONFIG
        )
        test_advice = generate_advice(loaded_model, loaded_tokenizer, "每天熬夜到凌晨1点", CONFIG)
        print(f"【加载模型后最终生成建议】：{test_advice}")
        print("模型加载和推理测试成功！")
    except Exception as e:
        print(f"模型加载测试失败：{e}")