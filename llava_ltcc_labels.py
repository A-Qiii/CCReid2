import os
import sys
# --- 0. 核心防爆盘设置 ---
os.environ['MODELSCOPE_CACHE'] = '/root/autodl-tmp/model_cache'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'

# 防止dataset重名问题
# 1. 获取当前脚本所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 彻底清理 sys.path 中可能指向当前目录的干扰项
# 无论它是绝对路径还是相对路径的 ''
while current_dir in sys.path:
    sys.path.remove(current_dir)
while '' in sys.path:
    sys.path.remove('')


import glob
import json
import torch
from tqdm import tqdm
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type
)

# --- 1. 满血版加载 ---
model_type = ModelType.llava1_5_7b_instruct
model, tokenizer = get_model_tokenizer(model_type, torch.float16, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 512
model.generation_config.temperature = 0.0  
model.generation_config.do_sample = False
template_type = get_default_template_type(model_type)
template = get_template(template_type, tokenizer)

# --- 2. 路径配置 (已切换至 LTCC 物理路径) ---
dataset_path = '/root/autodl-tmp/CCReID/data/LTCC/train' 
output_json = '/root/autodl-tmp/CCReID/logs/ltcc_llava_fp16_labels.json'
# 兼容读取 jpg 和 png
image_list = glob.glob(os.path.join(dataset_path, '*.jpg')) + glob.glob(os.path.join(dataset_path, '*.png'))

# --- 3. 强化版 Prompt (完全保留你的防御逻辑) ---
query_template = """<image>
Analyze the person in the image. You MUST strictly separate biological/accessory features from clothing.

DEFINITIONS:
- Identity: Gender, body shape (thin/normal/heavy), hair (length/color), glasses, backpacks, bags.
- Clothing: Upper garment, lower garment, footwear.

EXAMPLE OUTPUT:
{
    "full_description": "A thin man with short black hair wearing a red jacket, black pants and black shoes, carrying a black backpack.",
    "identity_features": Analyze the identity characteristics described in the text, such as"Man, thin body shape, short black hair, carrying a black backpack",
    "clothing_features": Analyze the clothing characteristics described in the text, such as"Red jacket, black pants, black shoes"
}

Now, analyze the provided image and output ONLY the JSON object following the exact structure above. Do not output any conversational text.
"""

# --- 新增：断点续传状态加载 ---
results = {}
if os.path.exists(output_json):
    with open(output_json, 'r', encoding='utf-8') as f:
        try:
            results = json.load(f)
            print(f"检测到历史文件，已恢复 {len(results)} 条记录。")
        except json.JSONDecodeError:
            print("历史 JSON 损坏，从头开始...")

# --- 4. 批量推理测试 ---
for idx, img_path in enumerate(tqdm(image_list[:], desc="LLaVA 满血版特征剥离 (LTCC)")):
    img_name = os.path.basename(img_path)
    
    # 防线 0：如果该图片已经存在于字典中，直接跳过，绝不重复计算
    if img_name in results:
        continue
        
    try:
        response, _ = inference(model, template, query_template, images=[img_path])
        
        # 1. 清理外壳
        clean_json = response.strip()
        if clean_json.startswith("```"):
            clean_json = clean_json.split("```")[1].replace("json", "").strip()
            
        # 2. 核心修复：抹除 LLaVA 乱加的反斜杠
        clean_json = clean_json.replace('\\_', '_')
        
        data = json.loads(clean_json)
        
        # 3. 二次防护：强制拍平
        if isinstance(data.get("identity_features"), dict):
            data["identity_features"] = ", ".join([str(v) for v in data["identity_features"].values()])
        if isinstance(data.get("clothing_features"), dict):
            data["clothing_features"] = ", ".join([str(v) for v in data["clothing_features"].values()])

        # 4. 过滤照抄 Prompt
        parrot_identity = "Man, thin body shape, short black hair, carrying a black backpack"
        parrot_clothing = "Red jacket, black pants, black shoes"
        
        if data.get("identity_features", "").strip() == parrot_identity or data.get("clothing_features", "").strip() == parrot_clothing:
            raise ValueError("模型触发退化行为，原封不动照抄了Prompt模板")

        # 5. 继承你的 ID 提取法则 (完美兼容 LTCC)
        data["id"] = img_name.split('_')[0] 
        results[img_name] = data
        
    except Exception as e:
        print(f"\n图片 {img_name} 解析失败: {e}. 原始输出: {response}")

    # --- 新增：每 50 张图强制落盘，防止显存爆炸/断网丢失 ---
    if (idx + 1) % 50 == 0:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

# 全量结束后的最终落盘
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\nLTCC 语义标签提取完成！请检查 {output_json}")