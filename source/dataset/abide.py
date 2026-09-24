import torch
import h5py
import numpy as np
from omegaconf import DictConfig, open_dict
def load_abide2_data(cfg: DictConfig):
    with h5py.File(cfg.dataset.path, 'r') as f:
        final_timeseires1 = f["AAL-90_dytimeseries"][:]
        final_timeseires2 = f["HO-96_dytimeseries"][:]
        final_timeseires3 = f["CC200_dytimeseries"][:]


        labels = f["label"][:]  # 如果是字符串数组，再 decode
        age = f["age"][:]
        sex = f["sex"][:]
        VIQ = f["VIQ"][:]
        PIQ = f["PIQ"][:]
        site = np.array([s.decode('utf-8') for s in f["site"]],dtype='<U8')
        unique, inverse = np.unique(site, return_inverse=True)
        site_label = inverse.astype(np.int64)
        non_imaging = np.stack((age,sex,VIQ,PIQ), axis=1)

    final_timeseires1,final_timeseires2,final_timeseires3,non_imaging,labels = [torch.from_numpy(
        data).float() for data in (final_timeseires1,final_timeseires2, final_timeseires3, non_imaging,labels)]


    with (open_dict(cfg)):
        cfg.dataset.node_sz = [final_timeseires1.shape[2], final_timeseires2.shape[2],final_timeseires3.shape[2]]
        cfg.dataset.node_feature_sz = [final_timeseires1.shape[3], final_timeseires2.shape[3],final_timeseires3.shape[3]]
        cfg.dataset.timeseries_sz = final_timeseires1.shape[3]
        cfg.dataset.windows_sz = final_timeseires1.shape[1]
        cfg.dataset.non_imaging_feature_sz = non_imaging.shape[1]
        cfg.dataset.atlas_name = ['AAL-90','HO-96','CC200']
    return final_timeseires1, final_timeseires2, final_timeseires3,non_imaging, labels, site_label

def load_adhd_data(cfg: DictConfig):
    with h5py.File(cfg.dataset.path, 'r') as f:
        final_timeseires1 = f["AAL-90_dytimeseries"][:]
        final_timeseires2 = f["HO-96_dytimeseries"][:]
        final_timeseires3 = f["CC200_dytimeseries"][:]

        labels = f["label"][:]
        age = f["age"][:]
        sex = f["sex"][:]
        VIQ = f["VIQ"][:]
        PIQ = f["PIQ"][:]
        site = np.array([s.decode('utf-8') for s in f["site"]],dtype='<U8')
        unique, inverse = np.unique(site, return_inverse=True)
        site_label = inverse.astype(np.int64)
        non_imaging = np.stack((age,sex,VIQ,PIQ), axis=1)

    final_timeseires1,final_timeseires2,final_timeseires3,non_imaging,labels = [torch.from_numpy(
        data).float() for data in (final_timeseires1,final_timeseires2, final_timeseires3, non_imaging,labels)]

    with (open_dict(cfg)):
        cfg.dataset.node_sz = [final_timeseires1.shape[2], final_timeseires2.shape[2],final_timeseires3.shape[2]]
        cfg.dataset.node_feature_sz = [final_timeseires1.shape[3], final_timeseires2.shape[3],final_timeseires3.shape[3]]
        cfg.dataset.timeseries_sz = final_timeseires1.shape[3]
        cfg.dataset.windows_sz = final_timeseires1.shape[1]
        cfg.dataset.non_imaging_feature_sz = non_imaging.shape[1]
        cfg.dataset.atlas_name = ['AAL-90','HO-96','CC200']
    return final_timeseires1, final_timeseires2, final_timeseires3,non_imaging, labels, site_label
