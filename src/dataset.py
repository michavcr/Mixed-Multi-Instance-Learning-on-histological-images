import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedKFold

class FeaturesDataset(Dataset):
    def __init__(self, dataset_dir, output_file=None, annotation_file=None):
        self.output_file = output_file
        self.dataset_dir = dataset_dir
        
        self.filenames = pd.Series(sorted(glob.glob(os.path.join(self.dataset_dir,'*'))))
        
        self.has_labels = (output_file is not None)
        self.has_annotations = (annotation_file is not None)
        
        if self.has_labels:
            self.output = pd.read_csv(output_file)
            self.labels = pd.to_numeric(self.output['Target'], errors='coerce')
        
        if self.has_annotations:
            annotations = pd.read_csv(annotation_file)
            annotations['ID_slide'] = annotations['Unnamed: 0'].apply(lambda s: s.split('.')[0].split('_')[1])
            annotations['ID_tile'] = annotations['Unnamed: 0'].apply(lambda s: s.split('.')[0].split('_')[4])
            annotations['coords_tile'] = annotations['Unnamed: 0'].apply(lambda s: s.split('.')[0].split('_')[5:8])
        
            self.annotations = annotations
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = self.filenames.iloc[idx]
        feat = np.load(path)[:,3:]
        
        if self.has_labels:
            label = self.labels.iloc[idx]
                
        if 'annotated' in path:
            id_slide = path.split('/')[-1].split('_')[1]
            mask = (self.annotations['ID_slide'] == id_slide)
            tile_annotations = self.annotations[mask]['Target']
            
            tile_annotations = tile_annotations.to_numpy().astype(np.float32)
            
            return feat.astype(np.float32), label.astype(np.float32), 1, tile_annotations
        
        if not self.has_labels:
            return feat.astype(np.float32), -1, -1, -1
        
        return feat.astype(np.float32), label.astype(np.float32), 0, np.array([])

def compute_dataset_stats(dataset, name='train'):
    """Compute some insightful stats about a dataset.

    Args:
        dataset (Dataset object): Dataset.
        name (str, optional): Name of the dataset. Defaults to 'train'.
    """
    # First, count the number of entries in the dataset
    N=dataset.__len__()
    N_pos, N_neg = 0,0
    N_annot, N_annot_pos = 0, 0
    
    N_feats = []
    print('There are {} entries in the {} dataset'.format(N, name))
    
    for i in range(N):
        feat, label, is_annot, _ = dataset.__getitem__(i)

        # Count the number of positive and negative labels
        if label:
            N_pos += 1
        else:
            N_neg += 1
        
        # Count the number of tiles per slide
        N_feats.append(feat.shape[0])
        
        # Count the number of slides that are annotated
        if is_annot:
            N_annot += 1
            if label:
                N_annot_pos +=1
                
    r_pos, r_neg = N_pos/N, N_neg/N
    r_annot = N_annot/N
    r_annot_pos = 0
    
    if N_annot>0:
        r_annot_pos = N_annot_pos/N_annot
        
    print('There are {} ({:.3f}%) positive examples and {} ({:.3f}%) negative examples in the {} dataset'.format(N_pos, r_pos*100, N_neg, r_neg*100, name))
    print('{} ({:.3f}%) slides are annotated. {} ({:.3f}%) are positive.'.format(N_annot, r_annot*100, N_annot_pos, r_annot_pos*100))
    
    plt.hist(N_feats, 10)
    plt.xlabel('Number of patches')
    plt.ylabel('Number of slides')
    plt.show()

def split_for_crossval(dataset, n_splits=5):
    """Split the dataset in n_splits balanced parts, in order to perform cross_validation.

    Args:
        dataset (Dataset object): Dataset to split.
        n_splits (int, optional): Number of folds. Defaults to 5.

    Returns:
        folds (List of (str, Dataset) dicts): List of (train set, validation set) couples (as dictionaries).
    """
    
    # StratifiedKFold to keep the class repartition accross the folds
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X = np.arange(dataset.__len__())
    y = dataset.labels

    folds=[]

    for i, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        td = Subset(dataset, train_idx)
        vd = Subset(dataset, val_idx)
        folds.append({'train': td, 'val': vd})
    
    return folds

