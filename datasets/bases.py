import torch
from torch.utils.data import Dataset
from PIL import Image
import os

def read_image(img_path):
    got_img = False
    if not os.path.exists(img_path):
        raise IOError(f"{img_path} 不存在")
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print(f"读取图片失败: {img_path}")
            pass
    return img

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, llava_dict=None):
        self.dataset = dataset
        self.transform = transform
        self.llava_dict = llava_dict if llava_dict is not None else {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_tuple = self.dataset[index]
        img_path = data_tuple[0]
        pid = data_tuple[1]
        camid = data_tuple[2]
        cloth_id = data_tuple[3] if len(data_tuple) > 3 and isinstance(data_tuple[3], int) else 0

        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        # 文本拼接逻辑
        img_name = os.path.basename(img_path)
        parent_dir = os.path.basename(os.path.dirname(img_path))
        prcc_key = f"{parent_dir}/{img_name}"
        info = self.llava_dict.get(prcc_key, self.llava_dict.get(img_name, {}))

        # 1. 组装身份长句 (Gender + Body Shape)
        gender = info.get("gender", "person")
        body = info.get("body_shape", "normal")
        id_text = f"A photo of a {gender} with {body} body shape."

        # 2. 组装衣着长句 (Upper + Lower + Shoes + Bag)
        u, l, s, b = info.get("upper_garment", "None"), info.get("lower_garment", "None"), \
                     info.get("shoes", "None"), info.get("backpack", "None")
        
        cloth_parts = []
        if u != "None": cloth_parts.append(f"wearing {u}")
        if l != "None": cloth_parts.append(f"wearing {l}")
        if s != "None": cloth_parts.append(f"wearing {s}")
        if b != "None": cloth_parts.append(f"carrying {b}")
        
        if cloth_parts:
            cloth_text = "A photo of a person " + ", ".join(cloth_parts) + "."
        else:
            cloth_text = "A photo of a person."

        # 恢复 I/O 契约：返回 7 个元素
        return img, pid, camid, cloth_id, 0, id_text, cloth_text