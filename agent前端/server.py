# server.py — Flask 3.x 兼容版
from flask import Flask, request, jsonify
from flask_cors import CORS
import agent

app = Flask(__name__)
CORS(app)

# 💡 启动时加载模型 —— 替代 before_first_request（Flask 3.x 移除）
ckpt = 'health_advice_model_best.ckpt'
tokenizer = 'health_tokenizer.json'
config = './exported_model/config.json'

print("正在加载模型，请稍候...")
MODEL, TOKENIZER, CONFIG = agent.load_model_for_inference(ckpt, tokenizer, config)
print("模型加载完成！")

@app.route('/api/advice', methods=['POST'])
def advice():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': '消息格式错误'}), 400

    text = data['message']
    try:
        reply = agent.generate_advice(MODEL, TOKENIZER, text, CONFIG)
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 开启线程支持并发
    app.run(host='0.0.0.0', port=5000, threaded=True)
