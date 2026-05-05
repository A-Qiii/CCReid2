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

# ============================================================
# 【可选】如果你已经有旧格式的 JSON（6字段格式），
# 可以直接运行转换函数，无需重新推理。
# 将下面的 CONVERT_EXISTING 改为 True，然后运行本脚本。
# ============================================================
CONVERT_EXISTING = True   # True=只做转换，False=重新推理+清洗

existing_json_path  = '/root/autodl-tmp/CCReID/logs/ltcc_qwen2_final_labels.json'  # 旧格式 JSON
final_output_json   = '/root/autodl-tmp/CCReID/data/ltcc_qwen2_final_labels.json'   # 新格式输出路径
raw_output_json     = '/root/autodl-tmp/CCReID/logs/ltcc_qwen2_raw_labels_v2.json'  # 重新推理时的原始输出


def convert_old_to_new(old_json_path, new_json_path):
    """
    将旧格式（6字段）转换为新格式（双字段），
    与 PRCC 的 JSON 格式完全对齐，供 bases.py 统一读取。

    旧格式（每条记录）：
    {
        "id": "001",
        "full_description": "...",
        "gender": "Man",
        "body_shape": "Thin",
        "upper_garment": "Red jacket",
        "lower_garment": "Blue jeans",
        "shoes": "White sneakers",
        "backpack": "None"
    }

    新格式（每条记录）：
    {
        "id": "001",
        "identity_features": "Man, thin body shape",
        "clothing_features": "Red jacket, blue jeans, white sneakers"
    }
    """
    print(f">>> 正在加载旧格式 JSON: {old_json_path}")
    with open(old_json_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)

    print(f">>> 共 {len(old_data)} 条记录，开始转换格式...")

    def force_none(val):
        """将各种'无意义'的值统一归零"""
        v = str(val).lower().strip()
        none_keywords = ['none', 'unknown', 'invisible', 'null', 'n/a', '']
        noise_phrases = ['not visible', 'not present', 'not carrying',
                         'not wearing', 'unseen', 'no bag', 'no backpack',
                         'no specific', 'no visible', 'cannot be determined']
        if v in none_keywords:
            return None
        for phrase in noise_phrases:
            if phrase in v:
                return None
        return str(val).strip()

    # 第一步：收集每个 PID 的性别投票（多数决）
    id_gender_votes = {}
    for img_name, data in old_data.items():
        pid = data.get('id', '')
        g_raw = str(data.get('gender', '')).lower()
        if any(w in g_raw for w in ['woman', 'female', 'girl']):
            g = 'Woman'
        elif any(w in g_raw for w in ['man', 'male', 'boy']):
            g = 'Man'
        else:
            g = None
        if g:
            id_gender_votes.setdefault(pid, []).append(g)

    final_genders = {
        pid: Counter(votes).most_common(1)[0][0]
        for pid, votes in id_gender_votes.items()
        if votes
    }

    # 第二步：逐条构建新格式
    new_data = {}
    for img_name, data in old_data.items():
        pid = data.get('id', '')
        gender = final_genders.get(pid, 'Person')

        # 组装 identity_features
        body   = force_none(data.get('body_shape', ''))
        backpack = force_none(data.get('backpack', ''))

        id_parts = [gender]
        if body:
            id_parts.append(f"{body} body shape")
        if backpack:
            id_parts.append(f"carrying {backpack}")
        identity_features = ", ".join(id_parts)

        # 组装 clothing_features
        upper = force_none(data.get('upper_garment', ''))
        lower = force_none(data.get('lower_garment', ''))
        shoes = force_none(data.get('shoes', ''))

        cloth_parts = [p for p in [upper, lower, shoes] if p]
        clothing_features = ", ".join(cloth_parts) if cloth_parts else "clothes"

        new_data[img_name] = {
            "id": pid,
            "identity_features": identity_features,
            "clothing_features": clothing_features
        }

    # 落盘
    os.makedirs(os.path.dirname(new_json_path), exist_ok=True)
    with open(new_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

    # 打印几条样例验证
    print(f"\n>>> 转换完成！共 {len(new_data)} 条记录，已保存至: {new_json_path}")
    print("\n--- 随机抽检 3 条转换结果 ---")
    sample_keys = list(new_data.keys())[:3]
    for k in sample_keys:
        print(f"\n  图片: {k}")
        print(f"  identity_features : {new_data[k]['identity_features']}")
        print(f"  clothing_features : {new_data[k]['clothing_features']}")


# ============================================================
# 如果 CONVERT_EXISTING=True，只做转换，直接退出
# ============================================================
if CONVERT_EXISTING:
    if not os.path.exists(existing_json_path):
        print(f"[错误] 找不到旧格式 JSON: {existing_json_path}")
        print("请将 CONVERT_EXISTING 改为 False，重新推理生成。")
    else:
        convert_old_to_new(existing_json_path, final_output_json)
    exit(0)


# ============================================================
# 以下为重新推理流程（CONVERT_EXISTING=False 时执行）
# ============================================================
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type
)

print(">>> [阶段一] 启动 Qwen2-VL-7B 视觉解析引擎（双字段格式）...")

model_type = ModelType.qwen2_vl_7b_instruct
model, tokenizer = get_model_tokenizer(
    model_type, torch.bfloat16, model_kwargs={'device_map': 'auto'}
)
model.generation_config.max_new_tokens = 512
model.generation_config.temperature = 0.0
model.generation_config.do_sample = False
template_type = get_default_template_type(model_type)
template = get_template(template_type, tokenizer)

dataset_path = '/root/autodl-tmp/CCReID/data/LTCC/train'
image_list = (glob.glob(os.path.join(dataset_path, '*.jpg')) +
              glob.glob(os.path.join(dataset_path, '*.png')))

# 与 PRCC 完全对齐的双字段 Prompt
query_template = """<image>
Analyze the person in the image. You MUST strictly separate biological/accessory features from clothing.

DEFINITIONS:
- Identity: Gender, body shape (thin/normal/heavy), hair (length/color), glasses, backpacks, bags.
- Clothing: Upper garment, lower garment, footwear.

RULES:
- If any identity or clothing feature is invisible or absent, omit it entirely.
- Do NOT guess occluded features.

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
        except Exception:
            print(">>> 历史文件损坏，重新开始提取...")

for idx, img_path in enumerate(tqdm(image_list, desc="Qwen2-VL 视觉解析 (LTCC)")):
    img_name = os.path.basename(img_path)  # LTCC 所有图在同一目录，纯文件名是唯一主键
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

        # 拍平：如果 VLM 返回了 dict 类型的字段，强制转为字符串
        for key in ['identity_features', 'clothing_features']:
            if isinstance(data.get(key), dict):
                data[key] = ", ".join(str(v) for v in data[key].values())

        # 过滤照抄 Prompt 的退化行为
        parrot_id   = "Man, thin body shape, short black hair, carrying a black backpack"
        parrot_clo  = "Red jacket, black pants, black shoes"
        if (data.get("identity_features", "").strip() == parrot_id or
                data.get("clothing_features", "").strip() == parrot_clo):
            raise ValueError("模型照抄了 Prompt 模板，跳过该图片")

        data["id"] = img_name.split('_')[0]
        results[img_name] = data

    except Exception as e:
        print(f"\n[警告] 图片 {img_name} 提取崩溃，已跳过。错误: {e}")

    if (idx + 1) % 50 == 0:
        with open(raw_output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

with open(raw_output_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
print(f">>> 原始推理结果已保存: {raw_output_json}")

# ==========================================
# 阶段二：清洗与全局多数投票纠偏
# ==========================================
print("\n>>> [阶段二] 启动数据清洗...")

blacklist = ['shirt', 'pant', 'shoe', 'jacket', 'coat', 'short',
             'skirt', 'dress', 'clothing', 'wear', 'sleeve', 'jeans',
             'trouser', 'sweater', 'hoodie', 'blouse', 'vest']
noise_list = ['unknown', 'no specific', 'no visible', 'no accessories',
              'no glasses', 'no backpack', 'no bag', 'no hair',
              'not visible', 'cannot be']

id_gender_votes = {}
for img_name, data in results.items():
    pid  = data.get('id', '')
    idf  = str(data.get('identity_features', '')).lower()
    if any(w in idf for w in ['woman', 'female', 'girl']):
        g = 'Woman'
    elif any(w in idf for w in ['man', 'male', 'boy']):
        g = 'Man'
    else:
        g = None
    if g:
        id_gender_votes.setdefault(pid, []).append(g)

final_genders = {
    pid: Counter(votes).most_common(1)[0][0]
    for pid, votes in id_gender_votes.items()
    if votes
}

clean_results = {}
for img_name, data in results.items():
    pid      = data.get('id', '')
    raw_idf  = str(data.get('identity_features', ''))

    chunks     = [c.strip() for c in raw_idf.split(',')]
    safe_chunks = []
    for chunk in chunks:
        cl = chunk.lower()
        if any(bw in cl for bw in blacklist):
            continue
        if any(nw in cl for nw in noise_list):
            continue
        if cl in ['man', 'woman', 'male', 'female', 'person']:
            continue
        if chunk:
            safe_chunks.append(chunk)

    true_gender = final_genders.get(pid, 'Person')
    identity_features = (f"{true_gender}, " + ", ".join(safe_chunks)
                         if safe_chunks else true_gender)

    clean_results[img_name] = {
        "id": pid,
        "identity_features": identity_features,
        "clothing_features": data.get("clothing_features", "clothes")
    }

os.makedirs(os.path.dirname(final_output_json), exist_ok=True)
with open(final_output_json, 'w', encoding='utf-8') as f:
    json.dump(clean_results, f, indent=4, ensure_ascii=False)

print(f">>> 处理完成！新格式 JSON 已保存至: {final_output_json}")