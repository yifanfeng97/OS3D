import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Dict
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import open3d as o3d
import torchvision.transforms as T

# Multi-view Modality
def fetch_img_list(path: Union[str, Path], n_view):
    all_filenames = sorted(list(Path(path).glob('image/h_*.jpg')))
    all_view = len(all_filenames)
    filenames = all_filenames[::all_view//n_view][:n_view]
    return filenames

def read_image(path_list: Union[List[str], List[Path]], augment=False, img_size=224):
    if augment:
        transform = T.Compose([
                T.RandomResizedCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
    else:
        transform = T.Compose([
                T.Resize(img_size),
                T.ToTensor(),
            ])
    imgs = [transform(Image.open(v).convert("RGB")) for v in path_list]
    imgs = torch.stack(imgs)
    return imgs


# Point Cloud Modality
def fetch_pt_path(path: Union[str, Path], n_pt):
    return Path(path) / 'pointcloud' / f'pt_{n_pt}.pts'

def read_pointcloud(path: Union[str, Path], augment=False):
    pt = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    pt = pt - np.expand_dims(np.mean(pt, axis=0), 0)  
    dist = np.max(np.sqrt(np.sum(pt ** 2, axis=1)), 0)
    pt = pt / dist  
    if augment:
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        pt = np.add(np.multiply(pt, xyz1), xyz2).astype('float32')
    pt = torch.from_numpy(pt.astype(np.float32))
    return pt.transpose(0, 1)


# Voxel Modality
def fetch_vox_path(path: Union[str, Path], d_vox):
    return Path(path) / 'voxel' / f'vox_{d_vox}.ply'

def read_voxel(path: Union[str, Path], d_vox, augment=False):
    vox_3d = o3d.io.read_voxel_grid(str(path))
    vox_idx = torch.from_numpy(np.array([v.grid_index - 1 for v in vox_3d.get_voxels()])).long()
    vox = torch.zeros((d_vox, d_vox, d_vox))
    vox[vox_idx[:,0], vox_idx[:,1], vox_idx[:,2]] = 1
    return vox.unsqueeze(0)


# Mulit-modal 3D Object
class MM3DO_dataset(Dataset):
    def __init__(self, data_root, obj_list, modality_cfg, augment=False):
        super().__init__()
        data_root = Path(data_root)
        self.augment = augment
        self.cfg = modality_cfg
        self.obj_list = obj_list
        self.label_idx_list = [obj['label_idx'] for obj in obj_list]
        self.label_list = [obj['label'] for obj in obj_list]
        self.n_class = len(set(self.label_idx_list))
    
    def __getitem__(self, index):
        cur_obj = self.obj_list[index]
        cur_path, cur_label_idx = cur_obj['path'], cur_obj['label_idx']
        ret = [cur_label_idx, ]
        for m, m_cfg in self.cfg.items():
            if m == 'image':
                img_list = fetch_img_list(cur_path, **m_cfg)
                data = read_image(img_list, augment=self.augment)
            elif m == 'pointcloud':
                path = fetch_pt_path(cur_path, **m_cfg)
                data = read_pointcloud(path, augment=self.augment)
            elif m == 'voxel':
                path = fetch_vox_path(cur_path, **m_cfg)
                data = read_voxel(path, **m_cfg, augment=self.augment)
            ret.append(data)
        return ret
    
    def __len__(self):
        return len(self.obj_list)

def add_label_idx(obj_list, label_set):
    for obj in obj_list:
        obj['label_idx'] = label_set.index(obj['label'])
    return obj_list

def get_retrieval_list(cfg_path, ret_path):
    ret_list, ret_path, unseen_label = [], Path(ret_path), []
    with open(cfg_path, 'r') as fp:
        for line in fp.readlines():
            obj_name, label_name = line.strip().split(',')
            ret_list.append({
                'path': str(ret_path / obj_name),
                'label': label_name
            })
            unseen_label.append(label_name)
    return ret_list, sorted(set(unseen_label))

def OS3DOR_Dataset(data_root, modality_cfg, query_path=None, target_path=None):
    data_root = Path(data_root)
    if query_path is None:
        query_path = data_root / "query.txt"
    if target_path is None:
        target_path = data_root / "target.txt"
    
    # Train Set
    train_list, seen_label = [], []
    for label_root in data_root.glob('train/*'):
        label_name = label_root.name
        for obj_path in label_root.glob('*'):
            train_list.append({
                'path': str(obj_path),
                'label': label_name
            })
        seen_label.append(label_name)
    seen_label = sorted(set(seen_label))

    # Retrieval Set
    query_list, query_label_set = get_retrieval_list(query_path, data_root / 'query')
    target_list, target_label_set = get_retrieval_list(target_path, data_root / 'target')
    assert query_label_set == target_label_set
    unseen_label = query_label_set

    # Add Label Index
    train_list = add_label_idx(train_list, seen_label)
    query_list = add_label_idx(query_list, unseen_label)
    target_list = add_label_idx(target_list, unseen_label)

    train_data = MM3DO_dataset(data_root, train_list, modality_cfg, augment=True)
    query_data = MM3DO_dataset(data_root, query_list, modality_cfg, augment=False)
    target_data = MM3DO_dataset(data_root, target_list, modality_cfg, augment=False)
    return train_data, query_data, target_data

if __name__ == '__main__':
    data_root = 'data/OS-ESB-core'
    modality_cfg = {
        'image': {'n_view': 8},
        # 'pointcloud': {'n_pt': 1024},
        # 'voxel': {'d_vox': 32}
    }

    # img_root = Path('/media/fengyifan/本地磁盘/NTU/NTU_2000_MM/chess/Y3813_pawn/image')
    # img_list = sorted([str(p) for p in img_root.glob('*.jpg')])[::4]
    # imgs = read_image(img_list, augment=True)
    # print(imgs.shape)
    
    # pt = read_pointcloud('/media/fengyifan/本地磁盘/NTU/NTU_2000_MM/chess/Y3813_pawn/pointcloud/pt_1024.pts')
    # print(pt.shape)

    # vox = read_voxel('/media/fengyifan/本地磁盘/NTU/NTU_2000_MM/chess/Y3813_pawn/voxel/vox_32.ply')
    # print(vox.shape)

    train_set, query_set, target_set = OS3DOR_Dataset(data_root, modality_cfg)
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    for idx, (lbl, sample) in enumerate(train_dataloader):
        print(lbl)
