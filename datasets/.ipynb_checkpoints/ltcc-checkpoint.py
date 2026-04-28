import os
import glob
import re
import json


class LTCC(object):
    def __init__(self, root='/root/autodl-tmp/CCReID/data', llava_json_path=None, **kwargs):
        # 定义文件目录
        self.dataset_dir = os.path.join(root, 'LTCC')
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'test')

        if not os.path.exists(self.train_dir):
            raise RuntimeError(f"找不到目录: {self.train_dir}")

        # 加载LLaVA文本先验库
        self.llava_dict = {}
        if llava_json_path and os.path.exists(llava_json_path):
            print(f">>> 成功加载 LLaVA 文本库: {llava_json_path}")
            with open(llava_json_path, 'r', encoding='utf-8') as f:
                self.llava_dict = json.load(f)
        else:
            print(">>> [警告] 未检测到 LLaVA 文本库，系统可能退化为纯视觉模型。")

        self.train = self._process_dir(self.train_dir, relabel=True)
        self.query = self._process_dir(self.query_dir, relabel=False)
        self.gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.num_train_pids = len(set([x[1] for x in self.train]))
        self.num_train_cams = len(set([x[2] for x in self.train]))
        self.num_train_vids = 1

    # 处理图集目录
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '*.png')) + glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_([-\d]+)_c(\d+)')

        all_pids = set()
        for img_path in img_paths:
            match = pattern.search(os.path.basename(img_path))
            if match:
                pid = int(match.group(1))
                if pid == -1: continue
                all_pids.add(pid)

        pid2label = {pid: i for i, pid in enumerate(sorted(list(all_pids)))}

        dataset = []
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            match = pattern.search(img_name)
            if not match: continue
            pid, cloth_id, camid = map(int, match.groups())
            if pid == -1: continue
            if relabel: pid = pid2label[pid]

            # 提取对应的 LLaVA 文本，如果没有则使用缺省占位符
            cloth_text = self.llava_dict.get(img_name, "A photo of a person")

            dataset.append((img_path, pid, camid, cloth_id, cloth_text))

        return dataset
