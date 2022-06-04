"""
Author: Fengyifan
Email: evanfeng97@qq.com
"""

import scipy
import scipy.spatial
import numpy as np


def eval_retrieval_metric(query_fts, target_fts, query_lbls, target_lbls, metric="cosine"):
    dist_mat = scipy.spatial.distance.cdist(query_fts, target_fts, "cosine")
    s_map = map_score(dist_mat, query_lbls, target_lbls)
    s_ndcg = ndcg_score(dist_mat, query_lbls, target_lbls)
    s_anmrr = anmrr_score(dist_mat, query_lbls, target_lbls)
    s_pr = pr(dist_mat, query_lbls, target_lbls)
    print(f"{'mAP':>10s}|{'NDCG@100':>10s}|{'ANMRR':>10s}|")
    print(f"{s_map:10.5f}|{s_ndcg:10.5f}|{s_anmrr:10.5f}|")
    print(f"pr curve: \n{s_pr}")
    return {
        'map': s_map,
        'ndcg@100': s_ndcg,
        'anmrr': s_anmrr,
        's_pr': s_pr
    }


################################### metric #######################################

def pr(dist_mat, lbl_a, lbl_b, top_k=1e9):
    n_a, n_b = dist_mat.shape
    top_k = min(top_k, n_b)
    s_idx = dist_mat.argsort()
    ans = []
    for i in range(n_a):
        cur_pr = [0]*11
        order = s_idx[i][:top_k]
        p_list, r_list = [], []
        truth = (lbl_a[i] == lbl_b[order])
        r_seen, r_max = 0, truth.sum()
        for j in range(top_k):
            if truth[j]:
                r_seen += 1
                r_list.append(r_seen / r_max)
                p_list.append(r_seen / (j + 1))
        if r_seen != 0:
            for ii in range(len(p_list)):
                p_list[ii] = max(p_list[ii:])
            r_list, p_list = np.array(r_list), np.array(p_list)
            for idx, t in enumerate(np.arange(0., 1.1, 0.1)):
                if np.sum(r_list >= t) != 0:
                    cur_pr[idx] = np.max(p_list[r_list >= t])
        ans.append(cur_pr)
    return np.array(ans).mean(0).tolist()

def map_score(dist_mat, lbl_a, lbl_b, top_k=1e9):
    n_a, n_b = dist_mat.shape
    top_k = min(top_k, n_b)
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        p_list = []
        r = 0
        for j in range(top_k):
            if lbl_a[i] == lbl_b[order[j]]:
                r += 1
                p_list.append(r / (j + 1))
        if r > 0:
            for ii in range(len(p_list)):
                p_list[ii] = max(p_list[ii:])
            res.append(np.array(p_list).mean())
        else:
            res.append(0)
    return np.mean(res)

def nn_score(dist_mat, lbl_a, lbl_b):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        if lbl_a[i] == lbl_b[order[0]]:
            res.append(1)
        else:
            res.append(0)
    return np.mean(res)

def ndcg_score(dist_mat, lbl_a, lbl_b, k=100):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, n_b + 2)))
        dcg = np.cumsum([1.0/np.log2(idx+2) if lbl_a[i] == lbl_b[item] else 0.0 for idx, item in enumerate(order)])
        ndcg = (dcg/idcg)[k-1]
        res.append(ndcg)
    return np.mean(res)

def anmrr_score(dist_mat, lbl_a, lbl_b):
    # NG: number of ground truth images (target images) per query (vector)
    n_a, n_b = dist_mat.shape
    lbl_a, lbl_b = np.array(lbl_a), np.array(lbl_b)
    NG = np.array([(lbl_a[i]==lbl_b).sum() for i in range(lbl_a.shape[0])])
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        cur_NG = NG[i]
        K = min(4*cur_NG, 2*NG.max())
        order = s_idx[i]
        ARR = np.sum([(idx+1)/cur_NG if lbl_a[i] == lbl_b[order[idx]] else (K+1)/cur_NG for idx in range(cur_NG)])
        MRR = ARR - 0.5*cur_NG - 0.5
        NMRR = MRR / (K - 0.5*cur_NG + 0.5)
        res.append(NMRR)
    return np.mean(res)
