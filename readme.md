# OS3D
This repository implatement the OS3D for the open-set 3DOR (3D object retrieval) task. The main code is motivated by the [Example Code](https://github.com/yifanfeng97/OS-MN40-Example) of the SHREC'22 Open-set 3D Object Retrieval. The OS3D is implemented by directly concatenating the features extracted by those modality-specify networks ([MVCNN](http://vis-www.cs.umass.edu/mvcnn/)(ResNet18 as the backbone) for the multi-view modality, [PointNet](http://stanford.edu/~rqi/pointnet/) for the point cloud modality, and [ShapeNets](https://3dshapenets.cs.princeton.edu/) for the voxel modality). Then, the concatenated features are adopted to evaluate the retrieval performance on those unknown categories of the retrieval set on the **OS-MN40-core**, **OS-NTU-core**, and **OS-ESB-core** datasets. Detailed performance can be found in the "[Open-set 3DOR Leaderboards](https://moon-lab.tech/os3dor)".

## Environment
- Python 3.9
- Pytorch 1.11.0
- Open3D 0.15.2

## Dataset Download
[Download Page](https://moon-lab.tech/os3dor)


## Evaluation Metric
The definition of mAP, NDCG, ANMRR can refer to this [book](https://www.sciencedirect.com/topics/computer-science/criterion-measure).
