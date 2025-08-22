import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集 (优化最终版)
    - 将辅助函数 save_emb 和 load_mm_emb 作为静态方法整合进类中。
    - 修复了多进程文件句柄问题，优化了IPC开销。
    """

    def __init__(self, data_dir, args):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id
        
        # 1. 存储文件路径，而不是直接打开文件句柄
        self.data_file_path = self.data_dir / "seq.jsonl"
        # 2. 初始化文件句柄为None，每个worker将创建自己的句柄
        self.data_file = None

        # 安全地加载一次性的文件
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        # 3. 通过类名调用静态方法来加载多模态 embedding
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _load_user_data(self, uid):
        # 4. 惰性加载文件句柄，确保每个worker有独立的句柄
        if self.data_file is None:
            self.data_file = open(self.data_file_path, 'rb')

        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        return json.loads(line)

    def _random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t
    
    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat_raw = [None] * (self.maxlen + 1)
        pos_feat_raw = [None] * (self.maxlen + 1)
        neg_feat_raw = [None] * (self.maxlen + 1)

        if not ext_user_sequence:
            ext_user_sequence.append((0, {}, 0, 0))

        nxt = ext_user_sequence[-1]
        idx = self.maxlen
        ts = {record[0] for record in ext_user_sequence if record[2] == 1 and record[0]}

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            
            seq_feat_raw[idx] = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type

            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat_raw[idx] = self.fill_missing_feat(next_feat, next_i)
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat_raw[idx] = self.fill_missing_feat(self.item_feat_dict.get(str(neg_id), {}), neg_id)

            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat_raw, pos_feat_raw, neg_feat_raw

    def __len__(self):
        return len(self.seq_offsets)

    def _init_feat_info(self):
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {
            'user_sparse': ['103', '104', '105', '109'],
            'item_sparse': ['100', '117', '111', '118', '101', '102', '119', '120', '114', '112', '121', '115', '122', '116'],
            'item_array': [],
            'user_array': ['106', '107', '108', '110'],
            'item_emb': self.mm_emb_ids,
            'user_continual': [],
            'item_continual': []
        }

        f_indexer = self.indexer.get('f', {})
        for feat_id in feat_types['user_sparse'] + feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(f_indexer.get(feat_id, []))
        for feat_id in feat_types['item_array'] + feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(f_indexer.get(feat_id, []))
        for feat_id in feat_types['user_continual'] + feat_types['item_continual']:
            feat_default_value[feat_id] = 0.0
        for feat_id in feat_types['item_emb']:
            if self.mm_emb_dict.get(feat_id):
                feat_default_value[feat_id] = np.zeros(next(iter(self.mm_emb_dict[feat_id].values())).shape[0], dtype=np.float32)

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        if feat is None: feat = {}
        filled_feat = feat.copy()
        all_feat_ids = {fid for f_type in self.feature_types.values() for fid in f_type}
        missing_fields = all_feat_ids - set(feat.keys())
        for feat_id in missing_fields:
            if feat_id in self.feature_default_value:
                filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types.get('item_emb', []):
            rev_id = self.indexer_i_rev.get(item_id)
            if rev_id and self.mm_emb_dict.get(feat_id) and rev_id in self.mm_emb_dict[feat_id]:
                emb = self.mm_emb_dict[feat_id][rev_id]
                filled_feat[feat_id] = np.array(emb, dtype=np.float32) if isinstance(emb, list) else emb
        return filled_feat

    @staticmethod
    def _collate_features_internal(feat_list, batch_size, seq_len):
        collated_feats = {}
        sample_feat_dict = next((feat for seq in feat_list for feat in seq if feat is not None), None)
        if not sample_feat_dict: return {}
        
        for key in sample_feat_dict.keys():
            is_array = isinstance(sample_feat_dict[key], (list, np.ndarray))
            
            flat_tensors = []
            for item_seq in feat_list:
                for step_feat in item_seq:
                    if step_feat and key in step_feat:
                        feat_val = step_feat[key]
                        tensor = torch.from_numpy(np.array(feat_val)) if is_array else torch.tensor(feat_val)
                        flat_tensors.append(tensor)
                    else:
                        default_val = [0] if is_array else 0
                        flat_tensors.append(torch.tensor(default_val))
            
            if is_array:
                padded = pad_sequence(flat_tensors, batch_first=True, padding_value=0)
                collated_feats[key] = padded.view(batch_size, seq_len, padded.shape[-1])
            else:
                stacked = torch.stack(flat_tensors)
                dtype = torch.float32 if isinstance(sample_feat_dict[key], float) else torch.long
                collated_feats[key] = stacked.view(batch_size, seq_len).to(dtype)
        return collated_feats

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)
        seq_len = len(batch[0][0])
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat_list, pos_feat_list, neg_feat_list = zip(*batch)

        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))

        seq_feat = MyDataset._collate_features_internal(seq_feat_list, batch_size, seq_len)
        pos_feat = MyDataset._collate_features_internal(pos_feat_list, batch_size, seq_len)
        neg_feat = MyDataset._collate_features_internal(neg_feat_list, batch_size, seq_len)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)
        # 覆盖训练集的文件路径
        self.data_file_path = self.data_dir / "predict_seq.jsonl"
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if isinstance(feat_value, list):
                processed_feat[feat_id] = [0 if isinstance(v, str) else v for v in feat_value]
            elif isinstance(feat_value, str):
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)
        user_id = "unknown"

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                user_id = u if isinstance(u, str) else self.indexer_u_rev.get(u, "unknown")
                if user_feat:
                    u_reid = 0 if isinstance(u, str) else u
                    ext_user_sequence.insert(0, (u_reid, self._process_cold_start_feat(user_feat), 2))
            if i and item_feat:
                i_reid = 0 if i > self.itemnum else i
                ext_user_sequence.append((i_reid, self._process_cold_start_feat(item_feat), 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat_raw = [None] * (self.maxlen + 1)
        idx = self.maxlen
        
        for record_tuple in reversed(ext_user_sequence):
            if idx < 0: break
            i, feat, type_ = record_tuple
            seq_feat_raw[idx] = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            idx -= 1
            
        return seq, token_type, seq_feat_raw, user_id

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)
        # 确保 batch 不为空
        if batch_size == 0:
            return None, None, None, []
        seq_len = len(batch[0][0])
        seq, token_type, seq_feat_list, user_id = zip(*batch)

        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        
        seq_feat = MyDataset._collate_features_internal(seq_feat_list, batch_size, seq_len)

        return seq, token_type, seq_feat, list(user_id)

def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict