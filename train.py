import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import json
import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader

from models import OS3D
from sklearn.metrics import f1_score
from datasets import OS3DOR_Dataset
from retrieval_metric import eval_retrieval_metric

######### must config this #########
dataname = 'MN40'
data_root = f'data/OS-{dataname}-core'
####################################

# configure
max_epoch = 60
modality_cfg = {
    'image': {'n_view': 2},
    'pointcloud': {'n_pt': 1024},
    'voxel': {'d_vox': 32}
}
this_task = f"OS-{dataname}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"

# log and checkpoint
out_dir = Path('cache')
save_dir = out_dir/'ckpts'/this_task
save_dir.mkdir(parents=True, exist_ok=True)

def setup_seed():
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"random seed: {seed}")

def train(data_loader, net, criterion, optimizer, epoch):
    print(f"Epoch {epoch}, Training...")
    net.train()
    loss_epoch = 0
    all_lbls, all_preds = [], []

    st = time.time()
    for i, (lbl, img, pt, vox) in enumerate(data_loader):
        img = img.cuda()
        pt = pt.cuda()
        vox = vox.cuda()
        lbl = lbl.cuda()
        data = (img, pt, vox)

        optimizer.zero_grad()
        out = net(data)
        out_img, out_pt, out_vox = out
        out_obj = (out_img + out_pt + out_vox)/4
        loss = criterion(out_obj, lbl)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(out_obj, 1)
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
        loss_epoch += loss.item()
        print(f"\t[{i}/{len(data_loader)}], Loss {loss.item()/lbl.shape[0]:.4f}")

    f1_micro = f1_score(all_lbls, all_preds, average="micro")
    f1_macro = f1_score(all_lbls, all_preds, average="macro")
    print(f"Epoch: {epoch}, Time: {time.time()-st:.4f}s, Loss: {loss_epoch/len(data_loader):4f}")
    print(f"{f1_micro=:.5f}, {f1_macro=:.5f}")
    print("This Epoch Done!\n")

@torch.no_grad()
def extract_ft_lbl(data_loader, net):
    net.eval()
    all_lbls = []
    fts_img, fts_pt, fts_vox = [], [], []
    for lbl, img, pt, vox in data_loader:
        img = img.cuda()
        pt = pt.cuda()
        vox = vox.cuda()
        lbl = lbl.cuda()
        data = (img, pt, vox)

        _, ft = net(data, global_ft=True)
        ft_img, ft_pt, ft_vox = ft

        all_lbls.append(lbl.detach().cpu().numpy())
        fts_img.append(ft_img.detach().cpu().numpy())
        fts_pt.append(ft_pt.detach().cpu().numpy())
        fts_vox.append(ft_vox.detach().cpu().numpy())

    all_lbls = np.concatenate(all_lbls, axis=0)
    fts_img = np.concatenate(fts_img, axis=0)
    fts_pt = np.concatenate(fts_pt, axis=0)
    fts_vox = np.concatenate(fts_vox, axis=0)
    all_fts = np.concatenate((fts_img, fts_pt, fts_vox), axis=1)
    return all_fts, all_lbls


def os_retrieve(query_loader, target_loader, net, epoch):
    print(f"{epoch=}, OSR evaluation...")
    st = time.time()
    q_fts, q_lbls = extract_ft_lbl(query_loader, net)
    t_fts, t_lbls = extract_ft_lbl(target_loader, net)
    res = eval_retrieval_metric(q_fts, t_fts, q_lbls, t_lbls)
    print("This Epoch Done!\n")
    return res


def save_checkpoint(net: nn.Module, res):
    state_dict = net.state_dict()
    ckpt = dict(
        net=state_dict,
        res=res,
    )
    torch.save(ckpt, str(save_dir / 'ckpt.pth'))
    with open(str(save_dir / 'ckpt.meta'), 'w') as fp:
        json.dump(res, fp)


def main():
    # init train_loader and val_loader
    setup_seed()
    print("Loader Initializing...\n")
    train_set, query_set, target_set = OS3DOR_Dataset(data_root, modality_cfg)
    print(f'train samples: {len(train_set)}')
    print(f'query samples: {len(query_set)}')
    print(f'target samples: {len(target_set)}')
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=16, drop_last=True)
    query_loader = DataLoader(query_set, batch_size=32, shuffle=False, num_workers=16)
    target_loader = DataLoader(target_set, batch_size=32, shuffle=False, num_workers=16)
    print("create new model")
    net = OS3D(train_set.n_class)
    net = net.cuda()
    net = nn.DataParallel(net)

    optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    best_res, best_state = None, 0
    for epoch in range(max_epoch):
        # train
        train(train_loader, net, criterion, optimizer, epoch)
        lr_scheduler.step()
        # validation
        # if epoch != 0 and epoch % 10 == 0:
        if epoch % 20 == 0:
            with torch.no_grad():
                res = os_retrieve(query_loader, target_loader, net, epoch)
            # save checkpoint
            if res['map'] > best_state:
                print("saving model...")
                best_res, best_state = res, res['map']
                save_checkpoint(net.module, res)

    print("\nTrain Finished!")
    print("Best 3DOR Results:")
    print(" | ".join(f"{k:>10s}" for k in best_res.keys() if k != 's_pr'))
    print(" | ".join(f"{v:10.5f}" for k, v in best_res.items()  if k != 's_pr'))
    print("pr curve:")
    print(", ".join( f"{v:.5f}" for v in best_res['s_pr']))
    print(f'checkpoint can be found in {save_dir}!')
    return best_res


if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
