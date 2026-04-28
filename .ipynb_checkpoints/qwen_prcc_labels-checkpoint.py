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
print(">>> [阶段一] 启动 Qwen2-VL-7B 视觉解析引擎...")

model_type = ModelType.qwen2_vl_7b_instruct
model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 512
model.generation_config.temperature = 0.0
model.generation_config.do_sample = False
template_type = get_default_template_type(model_type)
template = get_template(template_type, tokenizer)

# 锁定 PRCC 数据集的物理入口
dataset_path = '/root/autodl-tmp/CCReID/data/prcc/rgb/train' 
raw_output_json = '/root/autodl-tmp/CCReID/logs/prcc_qwen2_raw_labels.json'
final_output_json = '/root/autodl-tmp/CCReID/logs/prcc_qwen2_final_labels.json'

# 执行深层递归扫描，抓取所有层级下的图像
image_list = glob.glob(os.path.join(dataset_path, '**', '*.jpg'), recursive=True) + \
             glob.glob(os.path.join(dataset_path, '**', '*.png'), recursive=True)

query_template = """<image>
Analyze the person in the image. You MUST strictly separate biological/accessory features from clothing.

DEFINITIONS:
- Identity: Gender, body shape (thin/normal/heavy), hair (length/color), glasses, backpacks, bags.
- Clothing: Upper garment, lower garment, footwear.

EXAMPLE OUTPUT:
{
    "full_description": "A thin man with short black hair wearing a red jacket, black pants and black shoes, carrying a black backpack.",
    "identity_features": "Man, thin body shape, short black hair, carrying a black backpack",
    "clothing_features": "Red jacket, black pants, black shoes"
}

Now, analyze the provided image and output ONLY the JSON object following the exact structure above. Do not output any conversational text.
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
    # 【架构重构1】：截取相对路径作为 JSON 的唯一主键 (如 'test/A/060/img.jpg')
    rel_path = os.path.relpath(img_path, dataset_path).replace('\\', '/')
    
    if rel_path in results:
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
        
        # 【架构重构2】：利用目录树物理特性，提取上一级父文件夹名称作为 PID
        data["id"] = os.path.basename(os.path.dirname(img_path)) 
        results[rel_path] = data
        
    except Exception as e:
        print(f"\n[警告] 图片 {rel_path} 提取崩溃，已跳过。错误: {e}")

    if (idx + 1) % 50 == 0:
        with open(raw_output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

with open(raw_output_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# ==========================================
# 阶段二：物理清洗与全局多数投票纠偏
# ==========================================
print("\n>>> [阶段二] 启动全局数据清洗与物理校验...")

# 1. 衣服黑名单（维持原状）
blacklist = ['shirt', 'pant', 'shoe', 'jacket', 'coat', 'short', 'skirt', 'dress', 'clothing', 'wear', 'sleeve', 'jeans']

# 2. 【新增】语义噪音黑名单（专门剿灭废话）
noise_list = ['unknown', 'no specific', 'no visible', 'no accessories', 'no glasses', 'no backpack', 'no bag', 'no hair']

id_gender_votes = {}

for rel_path, data in results.items():
    pid = data['id']
    idf = str(data.get('identity_features', '')).lower()
    
    if 'woman' in idf or 'female' in idf or 'girl' in idf:
        gender = 'Woman'
    elif 'man' in idf or 'male' in idf or 'boy' in idf:
        gender = 'Man'
    else:
        gender = 'Unknown'
        
    if pid not in id_gender_votes:
        id_gender_votes[pid] = []
    if gender != 'Unknown':
        id_gender_votes[pid].append(gender)

final_genders = {}
for pid, votes in id_gender_votes.items():
    if votes:
        final_genders[pid] = Counter(votes).most_common(1)[0][0]
    else:
        final_genders[pid] = 'Person'

clean_results = {}

for rel_path, data in results.items():
    pid = data['id']
    raw_idf = str(data.get('identity_features', ''))
    
    chunks = [c.strip() for c in raw_idf.split(',')]
    safe_chunks = []
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        
        # 拦截逻辑 1：拦截衣服词汇
        has_clothes = any(bad_word in chunk_lower for bad_word in blacklist)
        
        # 拦截逻辑 2：【新增】拦截未知/无意义的废话
        has_noise = any(noise_word in chunk_lower for noise_word in noise_list)
        
        if not has_clothes and not has_noise:
            # 过滤掉单独的性别词，由后面的全局投票接管
            if chunk_lower not in ['man', 'woman', 'male', 'female', 'person']:
                safe_chunks.append(chunk)
                
    true_gender = final_genders.get(pid, 'Person')
    
    # 重组干净的文本
    if safe_chunks:
        purified_idf = f"{true_gender}, " + ", ".join(safe_chunks)
    else:
        purified_idf = true_gender  # 如果全被过滤了，至少保留一个绝对正确的性别锚点
        
    data['identity_features'] = purified_idf
    clean_results[rel_path] = data

with open(final_output_json, 'w', encoding='utf-8') as f:
    json.dump(clean_results, f, indent=4, ensure_ascii=False)

print(f">>> 物理清洗完成！完美数据已落盘至: {final_output_json}")