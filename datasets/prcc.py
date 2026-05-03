import os
import glob
import json

class PRCC(object):
    def __init__(self, root, llava_json_path=None, **kwargs):
        # 锁定 rgb 物理根目录
        self.dataset_dir = os.path.join(root, 'prcc', 'rgb')
        
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError(f"未找到 PRCC 根目录: {self.dataset_dir}")
            
        self.llava_dict = {}
        if llava_json_path and os.path.exists(llava_json_path):
            with open(llava_json_path, 'r', encoding='utf-8') as f:
                self.llava_dict = json.load(f)

        self.train = []
        self.query = []
        self.gallery = []

        # 启动物理目录推土机
        self._process_physical_folders()
        
        # 老老实实计算 Dataloader 需要的统计信息
        self.num_train_pids, self.num_train_cams, self.num_train_clothes = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_cams, self.num_query_clothes = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_cams, self.num_gallery_clothes = self.get_imagedata_info(self.gallery)
        
        print(f"=> PRCC 数据集物理加载完毕")
        print(f"   Train: {len(self.train)} 张 | Query: {len(self.query)} 张 | Gallery: {len(self.gallery)} 张")
        print(f"   Train PIDs: {self.num_train_pids} | Cams: {self.num_train_cams} | Clothes: {self.num_train_clothes}")

    def get_imagedata_info(self, data):
        # 遍历数据，用集合(set)提取出不重复的 PID, CAM 和 CLOTH_ID 数量
        pids, cams, clothes = set(), set(), set()
        for _, pid, camid, cloth_id, _, _ in data:
            pids.add(pid)
            cams.add(camid)
            clothes.add(cloth_id)
        return len(pids), len(cams), len(clothes)

    def _process_physical_folders(self):
        cam_map = {'A': 0, 'B': 1, 'C': 2}
        
        # ==========================================================
        # 1. 训练集解析与连续标签映射 (Label Encoding)
        # ==========================================================
        train_paths = glob.glob(os.path.join(self.dataset_dir, 'train', '*', '*.jpg'))
        
        # 步骤 1.1: 提取所有独立的原始 PID
        train_pids = set()
        for img_path in train_paths:
            normalized_path = img_path.replace("\\", "/")
            pid_str = os.path.basename(os.path.dirname(normalized_path))
            train_pids.add(int(pid_str))
            
        # 步骤 1.2: 构建连续的映射字典 {raw_pid: continuous_label}
        pid2label = {pid: label for label, pid in enumerate(sorted(list(train_pids)))}
        
        # 步骤 1.3: 载入数据并应用映射，同时生成全局唯一的 cloth_id
        for img_path in train_paths:
            normalized_path = img_path.replace("\\", "/")
            
            pid_str = os.path.basename(os.path.dirname(normalized_path))
            raw_pid = int(pid_str)
            mapped_pid = pid2label[raw_pid]  # 应用连续整数映射
            
            filename = os.path.basename(normalized_path)
            camid_str = filename[0].upper()
            camid = cam_map.get(camid_str, 0)
            
            # 【核心突破】：根据 PRCC 协议生成连续且全局唯一的 cloth_id
            # 同一个人在 A/B 相机是同一套衣服，在 C 相机是另一套
            if camid_str in ['A', 'B']:
                cloth_id = mapped_pid * 2
            else:  # 'C'
                cloth_id = mapped_pid * 2 + 1
            
            # 提取 JSON 文本锚点
            text_key = f"{pid_str}/{filename}"
            text_info = self.llava_dict.get(text_key, {})
            id_text = text_info.get("identity_features", "a person")
            cloth_text = text_info.get("clothing_features", "clothes")
            
            # 严格打包为 6 元组
            self.train.append((normalized_path, mapped_pid, camid, cloth_id, id_text, cloth_text))

        # 步骤 1.4: 将稀疏 cloth_id（0,2,4...）重映射为连续整数（0,1,2...）
        # 原因：ctx_cloth 大小 = num_train_clothes（唯一数量），但稀疏 cloth_id 最大值远超此数
        all_cloth_ids = sorted(set(item[3] for item in self.train))
        cloth2label = {cid: i for i, cid in enumerate(all_cloth_ids)}
        self.train = [
            (item[0], item[1], item[2], cloth2label[item[3]], item[4], item[5])
            for item in self.train
        ]

        # ==========================================================
        # 2. 测试集解析 (保持原始 PID，用于距离度量算法)
        # ==========================================================
        for split in ['test', 'val']:
            split_paths = glob.glob(os.path.join(self.dataset_dir, split, '*', '*', '*.jpg'))
            for img_path in split_paths:
                normalized_path = img_path.replace("\\", "/")
                parts = normalized_path.split("/")
                
                split_idx = parts.index(split)
                camid_str = parts[split_idx + 1].upper()
                camid = cam_map.get(camid_str, 0)
                raw_pid = int(parts[split_idx + 2])
                
                # 测试集不参与文本训练，赋予伪 cloth_id 和空白文本即可
                # 为了安全，给测试集 cloth_id 加上极大偏移，防止越界冲突
                cloth_id = raw_pid * 2 + (0 if camid_str in ['A', 'B'] else 1) + 10000 
                
                data_tuple = (normalized_path, raw_pid, camid, cloth_id, "", "")
                
                if camid_str == 'C':
                    self.query.append(data_tuple)
                else:
                    self.gallery.append(data_tuple)