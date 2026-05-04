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
    """
    统一数据集封装。

    JSON 格式约定（PRCC 与 LTCC 统一）：
    {
        "identity_features": "Man, thin body shape, short black hair",
        "clothing_features": "Red jacket, black pants, black shoes"
    }

    JSON 主键格式：
    - PRCC: "pid文件夹/文件名"，如 "003/A003_0001.jpg"
    - LTCC: 纯文件名，如 "001_01_c3.png"

    __getitem__ 返回 7 元组：
        img, pid, camid, cloth_id, 0, id_text, cloth_text
    """

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

        # -------------------------------------------------------
        # 文本读取：统一使用 identity_features / clothing_features
        # 优先以 "pid文件夹/文件名" 作为 key（兼容 PRCC）
        # 兜底以纯文件名作为 key（兼容 LTCC）
        # 最终兜底使用默认占位符
        # -------------------------------------------------------
        img_name = os.path.basename(img_path)
        parent_dir = os.path.basename(os.path.dirname(img_path))
        prcc_key = f"{parent_dir}/{img_name}"

        info = self.llava_dict.get(prcc_key, self.llava_dict.get(img_name, {}))

        id_text = info.get("identity_features", "a person")
        cloth_text = info.get("clothing_features", "clothes")

        return img, pid, camid, cloth_id, 0, id_text, cloth_text