import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import json
import re
import os
from tokenizers import Tokenizer

# ======================== 复用训练程序的核心常量/函数（必须保留） ========================
# 1. 固定特殊token映射（与训练时完全一致，不能改）
SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[SOS]": 2,
    "[EOS]": 3
}

# 2. 中文文本清洗函数（与训练时一致）
def clean_chinese_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\u4e00-\u9fff0-9，。！？；：""''（）【】《》、·…—]', '', text)
    return text[:100]

# 3. 分词结果可视化函数（可选，用于调试）
def visualize_tokenization(tokenizer, text, desc="分词结果"):
    print(f"\n===== {desc} =====")
    print(f"原始文本：{text}")
    encoded = tokenizer.encode(text)
    print(f"Token列表：{encoded.tokens}")
    print(f"ID列表：{encoded.ids}")
    print(f"分词长度：{len(encoded.ids)}")
    print("-" * 50)

# 4. 模型核心组件（完全复用训练时的定义，必须保留）
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

# 5. 加载模型和分词器的核心函数
def load_model_for_inference(ckpt_path, tokenizer_path, config_path):
    """
    加载训练好的模型和分词器
    :param ckpt_path: 模型权重文件路径（如health_advice_model_best.ckpt）
    :param tokenizer_path: 分词器文件路径（health_tokenizer.json）
    :param config_path: 配置文件路径（./exported_model/config.json）
    :return: 加载好的模型、分词器、配置
    """
    # 1. 加载配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 2. 加载分词器
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # 3. 初始化模型结构（与训练时一致）
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
    
    # 4. 加载模型权重
    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)  # 切换到推理模式
    
    return model, tokenizer, config

# 6. 生成养生建议的核心函数（交互核心）
def generate_advice(model, tokenizer, input_text, config):
    """
    输入健康问题/不良习惯，生成养生建议
    :param model: 加载好的模型
    :param tokenizer: 加载好的分词器
    :param input_text: 用户输入的文本（如"每天熬夜到凌晨1点"）
    :param config: 模型配置
    :return: 生成的养生建议文本
    """
    model.set_train(False)
    sos_id = SPECIAL_TOKENS["[SOS]"]
    eos_id = SPECIAL_TOKENS["[EOS]"]
    pad_id = SPECIAL_TOKENS["[PAD]"]
    dtype = ms.int32  # 与训练时的dtype一致
    
    # 步骤1：清洗和编码输入文本
    input_text = clean_chinese_text(input_text)
    visualize_tokenization(tokenizer, input_text, desc="输入文本分词")  # 可选：可视化分词
    
    src_encoded = tokenizer.encode(input_text)
    src_ids_clean = [id for id in src_encoded.ids if id not in [sos_id, eos_id, pad_id, SPECIAL_TOKENS["[UNK]"]]]
    src_ids = [sos_id] + src_ids_clean[:config["max_input_len"]-2] + [eos_id]
    src_pad_len = config["max_input_len"] - len(src_ids)
    src_ids += [pad_id] * src_pad_len
    src_mask = [True]*(len(src_ids)-src_pad_len) + [False]*src_pad_len
    
    # 转换为MindSpore张量
    src_ids = ms.Tensor([src_ids], dtype)
    src_mask = ms.Tensor([src_mask], ms.bool_)
    
    # 步骤2：自回归生成输出文本
    tgt_ids = ms.Tensor([[sos_id]], dtype)  # 初始化生成序列（以[SOS]开头）
    for step in range(config["max_output_len"]-1):
        if tgt_ids.shape[1] >= config["max_output_len"]:
            break
        
        # 构建目标mask
        tgt_mask = ms.Tensor([[True]*tgt_ids.shape[1]], ms.bool_)
        
        # 模型前向推理
        logits = model(src_ids, src_mask, tgt_ids, tgt_mask)
        
        # 取最后一个token的预测结果（argmax选概率最高的token）
        next_token = ops.argmax(logits[:, -1, :], dim=-1)
        next_token = next_token.unsqueeze(1).astype(dtype)
        
        # 拼接新token到生成序列
        tgt_ids = ops.concat([tgt_ids, next_token], axis=1)
        
        # 遇到[EOS]停止生成
        if next_token.asnumpy()[0,0] == eos_id:
            break
    
    # 步骤3：解码生成的token ID为文本（过滤特殊token）
    output_ids = tgt_ids.asnumpy()[0].tolist()
    output_ids = [id for id in output_ids if id not in [sos_id, eos_id, pad_id, SPECIAL_TOKENS["[UNK]"]]]
    advice = tokenizer.decode(output_ids)
    
    # 兜底：避免空输出
    if not advice:
        advice = "建议规律作息，合理饮食，适度运动，保持良好生活习惯。"
    
    return advice

# ======================== 实际交互逻辑 ========================
if __name__ == "__main__":
    # 1. 设置MindSpore上下文（与训练时一致）
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("CPU")
    
    # 2. 配置文件/模型/分词器路径（根据你的实际路径修改）
    CKPT_PATH = "health_advice_model_best.ckpt"  # 最优模型权重
    TOKENIZER_PATH = "health_tokenizer.json"     # 分词器文件
    CONFIG_PATH = "./exported_model/config.json" # 配置文件
    
    # 3. 校验文件是否存在
    for path in [CKPT_PATH, TOKENIZER_PATH, CONFIG_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在：{path}，请检查路径是否正确！")
    
    # 4. 加载模型和分词器
    print("正在加载模型和分词器...")
    model, tokenizer, config = load_model_for_inference(CKPT_PATH, TOKENIZER_PATH, CONFIG_PATH)
    print("模型加载完成！")
    
    # 5. 交互式对话（持续输入，生成建议）
    print("\n===== 养生建议生成器 =====")
    print("输入健康问题/不良习惯，按回车生成建议；输入'退出'结束程序")
    while True:
        user_input = input("\n你：")
        if user_input.lower() in ["退出", "q", "quit", "exit"]:
            print("程序结束！")
            break
        if not user_input.strip():
            print("请输入有效文本！")
            continue
        
        # 生成建议
        advice = generate_advice(model, tokenizer, user_input, config)
        print(f"模型：{advice}")