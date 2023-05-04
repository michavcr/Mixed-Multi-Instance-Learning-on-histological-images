import os
import glob

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import numpy as np

from model import AttentionLayer, DSAttentionLayer, MixedMILModel

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def average_predictions(model_paths, test_set, w=None):
    """For N models saved on the disk (N paths in model_paths),
    it averages the predictions on the test set to produce an ensemble model.
    
    weights can be provided to give more importance to some models.

    Args:
        model_paths (List of str): List of the model paths on the disk.
        test_set (Dataset): The test dataset
        w (List of float, optional): Weights of the model. Defaults to None. None equivalent to [1,...,1].
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = []
    
    if w is None:
        w = [1 for m in model_paths]
    
    w = torch.Tensor(w)
    w = w / w.sum()
    
    for model_path in model_paths:
        model = torch.load(model_path,map_location=torch.device('cpu'))
        model.eval()
        model = model.to(device)
        models.append(model)

    test_dl = DataLoader(test_set, batch_size=1, shuffle=False)
    test_pred = []
    detail_test_pred = []
    
    for i, (feat, _, _, _) in enumerate(tqdm(test_dl)):            
        feat = feat.to(device)
        outputs = torch.Tensor([m(feat)[0] for m in models])
        detail_test_pred.append(outputs)
        test_pred.append((outputs*w).sum().item())
    
    return(test_pred, detail_test_pred)

def load_models_and_average(model_dir, test_dataset, w=None):
    model_paths = sorted(glob.glob(os.path.join(model_dir, '*.pth')))

    if w is None:
        w = [1]*len(model_paths)

    test_pred, detail_test_pred = average_predictions(model_paths, test_dataset, w=w)
    
    return test_pred, detail_test_pred

def write_prediction_file(test_pred, test_feat_dir, output_path):
    test_id = pd.Series([p.split('/')[-1][3:6] for p in sorted(glob.glob(os.path.join(test_feat_dir,'*')))])
    output_test = pd.DataFrame({'ID':test_id, 'Target':test_pred})
    output_test.to_csv(output_path, index=False)

def topk_attention_maps(model, dataset, N, K=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    l = dataset.__len__()
    idx = torch.randperm(l)[:K]
    samples = Subset(dataset, idx.tolist())
    samples_dl = DataLoader(samples, batch_size=1, shuffle=False)

    scores = []
    labels = []

    for i, (feat, label, _, _) in enumerate(tqdm(samples_dl)):            
        feat = feat.to(device)
        att = model(feat)[2]
        labels.append(label.long().item())
        scores.append(torch.topk(att, 10).values.cpu().detach().numpy())

    scores = np.array(scores).squeeze()

    weights_counts = { 'top_{}'.format(i+1):scores.squeeze()[:,i] for i in range(K)}

    names = [ str(i.item()) for i in idx]
    width = 0.5

    fig, ax = plt.subplots()
    left = np.zeros(N)
    colormap = {1: ['darksalmon', 'tomato'],
                0: ['forestgreen', 'limegreen'],
                -1: ['hotpink', 'darkmagenta']}

    for i, (_, weight_count) in enumerate(weights_counts.items()):
        p = ax.barh(names, weight_count, width, color=[colormap[labels[j]][i%2] for j in range(N)], left=left)
        left += weight_count

    ax.set_title("Top-{} attention scores repartition on {} random slides\n(Read from the left to the right)".format(K, N))

    red_patch = mpatches.Patch(color='tomato', label='Class 1')
    green_patch = mpatches.Patch(color='forestgreen', label='Class 0')
    magenta_patch = mpatches.Patch(color='darkmagenta', label='Unknown class')

    ax.legend(handles=[red_patch, green_patch, magenta_patch], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fancybox=True)
    ax.set_xlabel('Attention scores')
    ax.set_ylabel('Slides')
    plt.show()