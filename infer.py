import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model import BaselineModel


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--maxlen", default=101, type=int)

    # Baseline Model construction
    parser.add_argument("--hidden_units", default=64, type=int)
    parser.add_argument("--num_blocks", default=4, type=int)
    parser.add_argument("--num_epochs", default=3, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--l2_emb", default=0.0, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--state_dict_path", default=None, type=str)
    parser.add_argument("--norm_first", action="store_true")

    # MMemb Feature ID
    parser.add_argument(
        "--mm_emb_id",
        nargs="+",
        default=["81"],
        type=str,
        choices=[str(s) for s in range(81, 87)],
    )

    args = parser.parse_args()

    return args


def load_emb(file_path):
    """
    从二进制文件中加载Embedding.
    """
    with open(file_path, "rb") as f:
        num_points, num_dimensions = struct.unpack("II", f.read(8))
        if Path(file_path).suffix == ".fbin":
            data = np.fromfile(f, dtype=np.float32)
        elif Path(file_path).suffix == ".u64bin":
            data = np.fromfile(f, dtype=np.uint64)
        else:
            raise TypeError("Unsupported file type")
    return data.reshape((num_points, num_dimensions))


def torch_ann_search(
    query_emb, item_emb, item_ids, top_k, batch_size=1024, device="cuda"
):
    """
    使用PyTorch进行批处理的ANN检索.
    """
    query_tensor = torch.from_numpy(query_emb).to(device)
    item_tensor = torch.from_numpy(item_emb).to(device)

    # L2-normalize embeddings for cosine similarity calculation via dot product
    query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
    item_tensor = torch.nn.functional.normalize(item_tensor, p=2, dim=1)

    all_top_k_indices = []

    for i in tqdm(
        range(0, query_tensor.size(0), batch_size), desc="PyTorch ANN Search"
    ):
        batch_queries = query_tensor[i : i + batch_size]

        # 计算点积 (等价于余弦相似度，因为向量已标准化)
        similarity_scores = torch.matmul(batch_queries, item_tensor.T)

        # 获取top-k结果
        _, top_k_indices = torch.topk(similarity_scores, k=top_k, dim=1)
        all_top_k_indices.append(top_k_indices.cpu().numpy())

    # 拼接所有批次的结果
    final_indices = np.vstack(all_top_k_indices)

    # 使用索引获取原始的item_ids
    top_k_item_ids = item_ids[final_indices]

    return top_k_item_ids.tolist()


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    """
    生产候选库item的id和embedding

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    EMB_SHAPE_DICT = {
        "81": 32,
        "82": 1024,
        "83": 3584,
        "84": 4096,
        "85": 3584,
        "86": 3584,
    }
    candidate_path = Path(os.environ.get("EVAL_DATA_PATH"), "predict_set.jsonl")
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    with open(candidate_path, "r") as f:
        for line in f:
            line = json.loads(line)
            # 读取item特征，并补充缺失值
            feature = line["features"]
            creative_id = line["creative_id"]
            retrieval_id = line["retrieval_id"]
            item_id = indexer[creative_id] if creative_id in indexer else 0
            missing_fields = set(
                feat_types["item_sparse"]
                + feat_types["item_array"]
                + feat_types["item_continual"]
            ) - set(feature.keys())
            feature = process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types["item_emb"]:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(
                        EMB_SHAPE_DICT[feat_id], dtype=np.float32
                    )

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 保存候选库的embedding和sid
    model.save_item_emb(
        item_ids, retrieval_ids, features, os.environ.get("EVAL_RESULT_PATH")
    )
    with open(
        Path(os.environ.get("EVAL_RESULT_PATH"), "retrive_id2creative_id.json"), "w"
    ) as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id


def infer():
    args = get_args()
    data_path = os.environ.get("EVAL_DATA_PATH")
    # 假设 MyTestDataset 已经根据训练加速部分进行了修改
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,  # can use multi-worker now
        collate_fn=test_dataset.collate_fn,
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = (
        test_dataset.feat_statistics,
        test_dataset.feature_types,
    )
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(
        args.device
    )
    model.eval()

    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    all_embs = []
    user_list = []
    with torch.no_grad():  # Inference mode
        for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            seq, token_type, seq_feat, user_id = batch
            seq = seq.to(args.device)
            # 假设predict方法也接受tensorized features
            # 如果MyTestDataset未修改，这里需要调整
            logits = model.predict(seq, seq_feat, token_type)
            all_embs.append(logits.cpu().numpy().astype(np.float32))
            user_list.extend(user_id)

    # 生成候选库的embedding 以及 id文件
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer["i"],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )
    all_embs = np.concatenate(all_embs, axis=0)
    query_path = Path(os.environ.get("EVAL_RESULT_PATH"), "query.fbin")
    save_emb(all_embs, query_path)

    # 1. 加载生成的embedding和id文件
    item_emb_path = Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin")
    item_id_path = Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin")

    item_embeddings = load_emb(item_emb_path)
    item_retrieval_ids = load_emb(item_id_path).flatten()  # [N,]
    query_embeddings = all_embs

    # 2. 执行PyTorch实现的ANN检索
    top100_retrieved_ids = torch_ann_search(
        query_embeddings,
        item_embeddings,
        item_retrieval_ids,
        top_k=10,  # 在这里设置需要的top-k值
        device=args.device,
    )

    # 3. 将检索到的retrieval_id转为creative_id
    top10s = []
    for retrieved_ids_for_one_user in tqdm(top100_retrieved_ids, desc="Mapping IDs"):
        creative_ids = [
            retrieve_id2creative_id.get(int(rid), 0)
            for rid in retrieved_ids_for_one_user
        ]
        top10s.append(creative_ids)

    return top10s, user_list
