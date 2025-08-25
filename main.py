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

def get_grad_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def move_batch_to_device(batch, device):
    """
    递归地将batch中的所有张量移动到指定的设备。
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    else:
        return batch

if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    
    data_path = os.environ.get('TRAIN_DATA_PATH')
    args = get_args()
    
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8, # 可以设置大于0的num_workers
        collate_fn=dataset.collate_fn,
        pin_memory=True 
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8, 
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
    
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # if hasattr(torch, 'compile'):
    #     print("Compiling the model...")
    #     model = torch.compile(model)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1
    
    no_decay_types = (torch.nn.RMSNorm, torch.nn.Embedding)
    decay_params = []
    no_decay_params = []
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            if param_name == "bias":
                no_decay_params.append(param)
            elif isinstance(module, no_decay_types):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": 1e-6},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=(0.9, 0.98)
    )
    def lr_lambda(current_step):
        warmup_steps = 1000
        total_steps = args.num_epochs * len(train_loader)
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / total_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # 在这里将整个batch移动到GPU
            batch = move_batch_to_device(batch, args.device)
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch

            pos_logits, neg_logits, log_feats, pos_embs, neg_embs = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            optimizer.zero_grad()
            
            indices = (next_token_type == 1) # 使用布尔索引
            infonce_loss_mask = (next_token_type == 1)
            
            infonce_loss = model.compute_infonce_loss(log_feats, pos_embs, neg_embs, infonce_loss_mask)

            loss = infonce_loss 
            
            grad_norm = get_grad_norm(model)
            current_lr = optimizer.param_groups[0]['lr']
            log_json = json.dumps({
                'global_step': global_step, 'infonce_loss': infonce_loss.item(),
                'lr': current_lr, 'grad_norm': grad_norm, 'epoch': epoch, 'time': time.time()
            })
            log_file.write(log_json + '\n'); log_file.flush(); print(log_json)
            writer.add_scalar('Loss/train_infonce', infonce_loss.item(), global_step)
            writer.add_scalar('Loss/train_total', loss.item(), global_step)
            writer.add_scalar('LR', current_lr, global_step)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            
        model.eval()
        valid_loss_sum = 0
        with torch.no_grad(): # 在验证阶段禁用梯度计算
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                # 同样移动验证集的batch
                batch = move_batch_to_device(batch, args.device)
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch

                pos_logits, neg_logits, log_feats, pos_embs, neg_embs = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                
                indices = (next_token_type == 1)
                infonce_loss_mask = (next_token_type == 1)
                
                infonce_loss = model.compute_infonce_loss(log_feats, pos_embs, neg_embs, infonce_loss_mask)
                loss = infonce_loss 
                valid_loss_sum += loss.item()

        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()