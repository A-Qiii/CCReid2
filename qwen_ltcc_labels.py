import os
import sys

# --- 0. 核心防爆盘设置 ---
os.environ['MODELSCOPE_CACHE'] = '/root/autodl-tmp/model_cache'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'

current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir in sys.path:
    sys.path.remove(current_dir)
while '' in sys.path:
    sys.path.remove('')

import glob
import json
from collections import Counter
from tqdm import tqdm
import torch
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type
)

# ==========================================
# 阶段一：高精度大模型视觉特征提取 (Qwen2-VL)
# ==========================================
print(">>> [阶段一] 启动 Qwen2-VL-7B 视觉解析引擎 (细粒度 CoT 模式)...")

model_type = ModelType.qwen2_vl_7b_instruct
model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 512
model.generation_config.temperature = 0.0  
model.generation_config.do_sample = False
template_type = get_default_template_type(model_type)
template = get_template(template_type, tokenizer)

dataset_path = '/root/autodl-tmp/CCReID/data/LTCC/train' 
raw_output_json = '/root/autodl-tmp/CCReID/logs/ltcc_qwen2_raw_labels.json'
final_output_json = '/root/autodl-tmp/CCReID/logs/ltcc_qwen2_final_labels.json'

image_list = glob.glob(os.path.join(dataset_path, '*.jpg')) + glob.glob(os.path.join(dataset_path, '*.png'))

# 【核心重构】：引入思维链 (CoT) 与强硬的 None 截断协议
query_template = """<image>
Analyze the person in the image step-by-step.

Step 1: Write a comprehensive full description of the person's identity and clothing.
Step 2: Extract 6 specific fine-grained features from your description.

RULES FOR EXTRACTION:
1. gender: "Man" or "Woman".
2. body_shape: "Thin", "Normal", or "Heavy".
3. upper_garment: Describe the shirt/jacket (e.g., "Red t-shirt").
4. lower_garment: Describe the pants/shorts/skirt (e.g., "Blue jeans").
5. shoes: Describe the footwear (e.g., "White sneakers").
6. backpack: Describe the backpack, bag, or handbag.

CRITICAL MASK RULE: If ANY of the above features is occluded, invisible, or not present in the image (e.g., the image is cut off and shoes are not visible, or the person is not carrying a bag), you MUST output exactly "None" for that field. Do not guess.

EXAMPLE OUTPUT FORMAT:
{
    "full_description": "A thin man with short black hair wearing a red jacket, blue jeans and white sneakers. He is not carrying any bag.",
    "gender": "Man",
    "body_shape": "Thin",
    "upper_garment": "Red jacket",
    "lower_garment": "Blue jeans",
    "shoes": "White sneakers",
    "backpack": "None"
}

Now, analyze the provided image and output ONLY the JSON object.
"""

results = {}
if os.path.exists(raw_output_json):
    with open(raw_output_json, 'r', encoding='utf-8') as f:
        try:
            results = json.load(f)
            print(f">>> 检测到历史断点，已恢复 {len(results)} 条原始记录。")
        except:
            print(">>> 历史文件损坏，重新开始提取...")

for idx, img_path in enumerate(tqdm(image_list[:], desc="Qwen2-VL 视觉解析")):
    img_name = os.path.basename(img_path)
    if img_name in results:
        continue
        
    try:
        response, _ = inference(model, template, query_template, images=[img_path])
        
        clean_json = response.strip()
        if clean_json.startswith("```"):
            clean_json = clean_json.split("```")[1]
            if clean_json.startswith("json"):
                clean_json = clean_json[4:]
        clean_json = clean_json.strip().replace('\\_', '_')
        
        data = json.loads(clean_json)
        data["id"] = img_name.split('_')[0] 
        results[img_name] = data
        
    except Exception as e:
        print(f"\n[警告] 图片 {img_name} 提取崩溃，已跳过。错误: {e}")

    if (idx + 1) % 50 == 0:
        with open(raw_output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

with open(raw_output_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# ==========================================
# 阶段二：细粒度特征物理清洗与绝对 None 值校验
# ==========================================
print("\n>>> [阶段二] 启动细粒度特征清洗与掩码对齐...")

id_gender_votes = {}

# 第 1 趟遍历：收集性别选票 (保持原有的客观物理校验)
for img_name, data in results.items():
    pid = data['id']
    gender_val = str(data.get('gender', '')).lower()
    
    if 'woman' in gender_val or 'female' in gender_val or 'girl' in gender_val:
        g = 'Woman'
    elif 'man' in gender_val or 'male' in gender_val or 'boy' in gender_val:
        g = 'Man'
    else:
        g = 'Unknown'
        
    if pid not in id_gender_votes:
        id_gender_votes[pid] = []
    if g != 'Unknown':
        id_gender_votes[pid].append(g)

final_genders = {}
for pid, votes in id_gender_votes.items():
    if votes:
        final_genders[pid] = Counter(votes).most_common(1)[0][0]
    else:
        final_genders[pid] = 'Person'

clean_results = {}

# 辅助函数：无情地粉碎一切模糊语义，将其归零为 "None"
def force_clean_to_none(val):
    v_lower = str(val).lower().strip()
    if v_lower in ['', 'none', 'unknown', 'invisible', 'null', 'n/a']:
        return "None"
    # 拦截 VLM 可能生成的废话长句
    if "not visible" in v_lower or "no " in v_lower or "not carrying" in v_lower or "unseen" in v_lower:
        return "None"
    return str(val).strip()

# 第 2 趟遍历：执行 6 维度的硬性清洗
for img_name, data in results.items():
    pid = data['id']
    
    # 1. 注入绝对性别
    clean_data = {
        "id": pid,
        "full_description": data.get("full_description", ""),
        "gender": final_genders.get(pid, 'Person')
    }
    
    # 2. 依次清洗剩余 5 个局部特征，任何废话均强制坍缩为 "None"
    feature_keys = ['body_shape', 'upper_garment', 'lower_garment', 'shoes', 'backpack']
    for key in feature_keys:
        clean_data[key] = force_clean_to_none(data.get(key, ''))
        
    clean_results[img_name] = clean_data

# 最终绝对纯净版数据落盘
with open(final_output_json, 'w', encoding='utf-8') as f:
    json.dump(clean_results, f, indent=4, ensure_ascii=False)

print(f">>> 处理完成！数据已保存至: {final_output_json}")