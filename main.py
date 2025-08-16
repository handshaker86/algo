import argparse
import json
import os
import time
from pathlib import Path
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


def inbatch_loss(user_emb, pos_emb, next_token_type, loss_type='cross_entropy'):
    """
    计算in-batch负样本loss

    Args:
        user_emb: [B, L, D]
        pos_emb: [B, L, D]
        next_token_type: [B, L]，token类型掩码，1表示item
        loss_type: str, 选择损失计算方式，'cross_entropy'或'softmax_max_diff'

    Returns:
        loss_inbatch: 计算得到的in-batch负样本loss
    """
    B, L, D = user_emb.shape

    # flatten
    user_emb_2d = user_emb.reshape(B * L, D)
    pos_emb_2d = pos_emb.reshape(B * L, D)

    # mask，筛选有效item位置
    mask_1d = (next_token_type == 1).reshape(B * L)
    user_emb_valid = user_emb_2d[mask_1d]
    pos_emb_valid = pos_emb_2d[mask_1d]

    # 计算inbatch logits
    logits = torch.matmul(user_emb_valid, pos_emb_valid.T)  # [N, N], N=有效位置数

    if loss_type == 'cross_entropy':
        labels = torch.arange(logits.size(0), device=user_emb.device)
        criterion = torch.nn.CrossEntropyLoss()
        loss_inbatch = criterion(logits, labels)

    elif loss_type == 'softmax_max_diff':
        # 对角线为正样本分数
        pos_scores = logits.diagonal()  # [N]

        # 先将对角线元素屏蔽（负无穷），再取每行最大负样本得分
        diag_mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
        neg_logits = logits.masked_fill(diag_mask, float('-inf'))
        max_neg_scores, _ = neg_logits.max(dim=1)  # [N]

        # 计算差值并softmax：loss = -log( exp(pos - max_neg) / sum(exp(pos - max_neg)) )
        # 实际等价于对pos - max_neg做log_softmax并取负均值
        diff = pos_scores - max_neg_scores
        loss_inbatch = -torch.log_softmax(diff, dim=0).mean()

    else:
        raise ValueError(f"Unknown loss_type {loss_type}")

    return loss_inbatch

def get_grad_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    triplet_criterion = torch.nn.TripletMarginLoss(margin=0.5, p=2)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    # 1. 优化器增加 weight_decay
    no_decay_types = (torch.nn.RMSNorm, torch.nn.Embedding)

    decay_params = []
    no_decay_params = []

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            
            # bias 一般不 decay
            if param_name == "bias":
                no_decay_params.append(param)
            # LayerNorm / Embedding 全部不 decay
            elif isinstance(module, no_decay_types):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    # 参数分组
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": 1e-6},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=(0.9, 0.98)
    )
    # 2. 新增学习率调度器（warmup示例）
    
    def lr_lambda(current_step):
        warmup_steps = 1000
        total_steps = args.num_epochs * len(train_loader)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / total_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            pos_logits, neg_logits, log_feats, pos_embs, neg_embs = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            optimizer.zero_grad()
            indices = np.where(next_token_type == 1)
            infonce_loss_mask = (next_token_type == 1).to(args.device) # padding mask and item/user mask
            # act_type_mask = (next_action_type == 1).to(args.device) # action type mask 去掉act type为0的部分
            # infonce_loss_mask = infonce_loss_mask & act_type_mask # 只保留act type为1的部分
            infonce_loss = model.compute_infonce_loss(log_feats, pos_embs, neg_embs, infonce_loss_mask)
            # loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            # loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            triplet_loss = triplet_criterion(
                log_feats[indices], pos_embs[indices], neg_embs[indices]
            )

            # 总损失
            loss = infonce_loss +  triplet_loss  # 0.1 是权重，可调
            # log_json = json.dumps(
            #     {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            # )
            # log_file.write(log_json + '\n')
            # log_file.flush()
            # print(log_json)

            # writer.add_scalar('Loss/train', loss.item(), global_step)
            grad_norm = get_grad_norm(model)
            current_lr = optimizer.param_groups[0]['lr']
            log_json = json.dumps(
                {
                    'global_step': global_step,
                    'infonce_loss': infonce_loss.item(),
                    'triplet_loss': triplet_loss.item(),
                    'loss_total': loss.item(),
                    'lr': current_lr,
                    'grad_norm': grad_norm,
                    'epoch': epoch,
                    'time': time.time()
                }
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            if math.isnan(loss.item()) or math.isinf(grad_norm) or math.isnan(grad_norm):
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"NaN grad in {name}")
                        if torch.isinf(param.grad).any():
                            print(f"Inf grad in {name}")
                    if torch.isnan(param).any():
                        print(f"NaN in {name}")
                    if torch.isinf(param).any():
                        print(f"Inf in {name}")

            writer.add_scalar('Loss/train_infonce', infonce_loss.item(), global_step)
            writer.add_scalar('Loss/train_triplet', triplet_loss.item(), global_step)
            writer.add_scalar('Loss/train_total', loss.item(), global_step)
            writer.add_scalar('LR', current_lr, global_step)
            global_step += 1

            loss.backward()
            # 这里加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() 
            # 这里调用学习率调度器step，完成warmup
            scheduler.step()
            
        model.eval()
        valid_loss_sum = 0
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            pos_logits, neg_logits, log_feats, pos_embs, neg_embs = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            indices = np.where(next_token_type == 1)
            # loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            # loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            infonce_loss_mask = (next_token_type == 1).to(args.device)
            # act_type_mask = (next_action_type == 1).to(args.device) # action type mask 去掉act type为0的部分
            # infonce_loss_mask = infonce_loss_mask & act_type_mask # 只保留act type为1的部分
            infonce_loss = model.compute_infonce_loss(log_feats, pos_embs, neg_embs, infonce_loss_mask)
            triplet_loss = triplet_criterion(
                log_feats[indices], pos_embs[indices], neg_embs[indices]
            )
            loss = infonce_loss + triplet_loss
            valid_loss_sum += loss.item()
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()