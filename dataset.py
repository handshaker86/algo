import json
import os
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集
    """

    def __init__(self, data_dir, args):
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_paths_and_offsets()
        self.data_file = None  # 将由每个 worker 独立打开
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _load_paths_and_offsets(self):
        """仅加载路径和偏移量，不打开文件"""
        self.data_file_path = self.data_dir / "seq.jsonl"
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """为每个 worker 安全地加载用户数据"""
        if self.data_file is None:
            self.data_file = open(self.data_file_path, 'rb')

        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error in PID {os.getpid()}: JSONDecodeError for UID {uid}. Line content (first 100 chars): {line[:100]}")
            raise e
        return data

    def _random_neq(self, l, r, s):
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def _prepare_features(self, feature_list):
        """
        将特征字典列表转换为特征 Numpy 数组的字典.
        遵循图示逻辑：数组特征在此阶段不进行内部填充.
        """
        prepared_feats = {}
        all_feat_ids = list(self.feature_default_value.keys())

        for feat_id in all_feat_ids:
            # 多模态向量特征 (已是规整的 numpy array)
            if feat_id in self.feature_types['item_emb']:
                emb_dim = self.feature_default_value[feat_id].shape[0]
                feat_array = np.zeros((len(feature_list), emb_dim), dtype=np.float32)
                for i, feat_dict in enumerate(feature_list):
                    if feat_dict and feat_id in feat_dict:
                        feat_array[i] = feat_dict[feat_id]
                prepared_feats[feat_id] = feat_array
            # 数组特征 (保持为变长列表，不在此处 padding)
            elif feat_id in self.feature_types['user_array'] or feat_id in self.feature_types['item_array']:
                # 使用 dtype=object 来容纳变长的 list
                feat_array = np.empty(len(feature_list), dtype=object)
                for i, feat_dict in enumerate(feature_list):
                    if feat_dict and feat_id in feat_dict and isinstance(feat_dict[feat_id], list):
                        feat_array[i] = feat_dict[feat_id]
                    else:
                        feat_array[i] = self.feature_default_value[feat_id] # 默认值，例如 [0]
                prepared_feats[feat_id] = feat_array
            # ID 及稀疏类别特征
            else:
                dtype = np.float32 if feat_id in self.feature_types.get('user_continual', []) or feat_id in self.feature_types.get('item_continual', []) else np.int64
                feat_array = np.zeros(len(feature_list), dtype=dtype)
                for i, feat_dict in enumerate(feature_list):
                     if feat_dict and feat_id in feat_dict:
                        feat_array[i] = feat_dict[feat_id]
                prepared_feats[feat_id] = feat_array

        return prepared_feats

    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        # 序列长度统一 padding 至 maxlen + 1
        seq = np.zeros(self.maxlen + 1, dtype=np.int32)
        pos = np.zeros(self.maxlen + 1, dtype=np.int32)
        neg = np.zeros(self.maxlen + 1, dtype=np.int32)
        token_type = np.zeros(self.maxlen + 1, dtype=np.int32)
        next_token_type = np.zeros(self.maxlen + 1, dtype=np.int32)
        next_action_type = np.zeros(self.maxlen + 1, dtype=np.int32)

        seq_feat_list = [None] * (self.maxlen + 1)
        pos_feat_list = [None] * (self.maxlen + 1)
        neg_feat_list = [None] * (self.maxlen + 1)

        ts = {rec[0] for rec in ext_user_sequence if rec[2] == 1 and rec[0]}
        
        idx = 0
        for i, record_tuple in enumerate(ext_user_sequence):
            if idx > self.maxlen:
                break

            item_id, feat, type_, act_type = record_tuple
            seq[idx] = item_id
            token_type[idx] = type_
            seq_feat_list[idx] = self.fill_missing_feat(feat, item_id)

            # 如果不是序列最后一个元素，则有下一个元素可以作为label
            if i < len(ext_user_sequence) - 1:
                next_i, next_feat, next_type, next_act_type = ext_user_sequence[i+1]

                next_token_type[idx] = next_type
                if next_act_type is not None:
                    next_action_type[idx] = next_act_type

                if next_type == 1 and next_i != 0:
                    pos[idx] = next_i
                    pos_feat_list[idx] = self.fill_missing_feat(next_feat, next_i)
                    neg_id = self._random_neq(1, self.itemnum + 1, ts)
                    neg[idx] = neg_id
                    neg_feat_list[idx] = self.fill_missing_feat(self.item_feat_dict.get(str(neg_id)), neg_id)

            idx += 1
        
        seq_feat = self._prepare_features(seq_feat_list)
        pos_feat = self._prepare_features(pos_feat_list)
        neg_feat = self._prepare_features(neg_feat_list)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

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

        all_sparse_feat = feat_types['user_sparse'] + feat_types['item_sparse']
        all_array_feat = feat_types['user_array'] + feat_types['item_array']
        all_continual_feat = feat_types['user_continual'] + feat_types['item_continual']

        for feat_id in all_sparse_feat:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'].get(feat_id, {}))
        for feat_id in all_array_feat:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'].get(feat_id, {}))
        for feat_id in all_continual_feat:
            feat_default_value[feat_id] = 0.0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32)

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        if feat is None:
            feat = {}
        filled_feat = feat.copy()

        all_feat_ids = self.feature_default_value.keys()
        missing_fields = set(all_feat_ids) - set(feat.keys())

        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev.get(item_id) in self.mm_emb_dict.get(feat_id, {}):
                emb = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]
                if isinstance(emb, np.ndarray):
                    filled_feat[feat_id] = emb

        return filled_feat
    

    @staticmethod
    def collate_feats(feat_list):
            collated = {}
            if not feat_list or not feat_list[0]: return collated
            
            keys = feat_list[0].keys()
            for key in keys:
                arrays = [d[key] for d in feat_list]
                
                # 按照图示逻辑，在 collate_fn 中处理数组特征的内部 padding
                if arrays[0].dtype == object:
                    max_inner_len = 0
                    for seq_of_lists in arrays: # 遍历 batch
                        for inner_list in seq_of_lists: # 遍历序列
                            if inner_list is not None:
                                max_inner_len = max(max_inner_len, len(inner_list))

                    batch_size = len(arrays)
                    seq_len = len(arrays[0])
                    padded_batch = np.zeros((batch_size, seq_len, max_inner_len), dtype=np.int64)

                    for b_idx, seq_of_lists in enumerate(arrays):
                        for s_idx, inner_list in enumerate(seq_of_lists):
                            if inner_list is not None:
                                padded_batch[b_idx, s_idx, :len(inner_list)] = inner_list
                    
                    collated[key] = torch.from_numpy(padded_batch)
                # 其他特征直接堆叠
                else:
                    collated[key] = torch.from_numpy(np.stack(arrays))
            return collated

    @staticmethod
    def collate_fn(batch):
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)

        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))

        seq_feat_collated = MyDataset.collate_feats(seq_feat)
        pos_feat_collated = MyDataset.collate_feats(pos_feat)
        neg_feat_collated = MyDataset.collate_feats(neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat_collated, pos_feat_collated, neg_feat_collated


class MyTestDataset(MyDataset):
    """
    测试数据集
    """
    def _load_paths_and_offsets(self):
        """为测试集加载正确的路径和偏移量"""
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

        ext_user_sequence = []
        user_id = None
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                user_id = u if isinstance(u, str) else self.indexer_u_rev.get(u, f"user_{u}")
                u_reid = 0 if isinstance(u, str) else u
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u_reid, user_feat, 2))
            if i and item_feat:
                i_reid = i if i <= self.itemnum else 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i_reid, item_feat, 1))

        seq = np.zeros(self.maxlen + 1, dtype=np.int32)
        token_type = np.zeros(self.maxlen + 1, dtype=np.int32)
        seq_feat_list = [None] * (self.maxlen + 1)
        
        idx = 0
        for record_tuple in ext_user_sequence:
            if idx > self.maxlen: break
            i, feat, type_ = record_tuple
            seq[idx] = i
            token_type[idx] = type_
            seq_feat_list[idx] = self.fill_missing_feat(feat, i)
            idx += 1

        seq_feat = self._prepare_features(seq_feat_list)
        return seq, token_type, seq_feat, user_id

    @staticmethod
    def collate_fn(batch):
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        
        collated_seq_feat = MyDataset.collate_feats(seq_feat)

        return seq, token_type, collated_seq_feat, user_id


def save_emb(emb, save_path):
    num_points, num_dimensions = emb.shape
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
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
                            insert_emb = np.array(data_dict_origin['emb'], dtype=np.float32)
                            emb_dict[data_dict_origin['anonymous_cid']] = insert_emb
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict