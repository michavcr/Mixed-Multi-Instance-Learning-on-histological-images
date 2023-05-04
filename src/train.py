import os
import glob

import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm 
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from dataset import compute_dataset_stats

def train_an_epoch(model, train_dl, criterion, optimizer, per, lambda1, lambda2, device):
    """Called by train at each epoch.

    Args:
        model (nn.Module object): model to train for an epoch
        train_dl (DataLoader object): train dataloader
        criterion (Loss object): loss function (used both for slides and tiles)
        optimizer (Optimizer object): optimizer
        per (float): percentage of pseudo-labels to produce from attention scores 
                    (if n non-annoted tiles, then K=[per*n] positive and K negative pseudo labels are
                     produced by keeping the tiles that have the top-K largest and smallest attention scores)
        lambda1 (float): in [0,1], weight for the slide loss
        lambda2 (float): in [0,1], weight for the tile loss
        device (str): device to run the calculations.

    Returns:
        train_loss, train_tile_loss, train_slide_loss, train_auc, train_tiles_auc: losses and AUCs at this epoch 
    """
    model.train()
    
    running_loss = 0
    running_tile_loss = 0
    running_slide_loss = 0

    train_truth = []
    train_pred = []
    train_tiles_truth = []
    train_tiles_probs = []

    for i, (feat, label, is_annot, tile_annot) in enumerate(tqdm(train_dl)):
        optimizer.zero_grad()
        feat, label, is_annot, tile_annot = feat.to(device), label.to(device), is_annot.to(device), tile_annot.to(device)

        output, tiles_output, att_scores = model(feat)
        
        tiles_output = tiles_output[0,:,0]
        att_scores = att_scores[0]

        if not label[0].item():
            # if the slide is labeled negative, then all the tiles shall be labeled negative
            tiles_labels = torch.zeros(tiles_output.shape, device=device)
            tiles_probs = tiles_output

        elif is_annot:
            # if the slide is labeled positive and has annotations, then just use them
            tiles_labels = tile_annot[0]
            tiles_probs = tiles_output

        else:
            # if the slide is labeled positive and has no annotation,
            # use the topk attention scores to produce pseudo-labels
            K = int(per * len(att_scores))
            pos_idx = torch.topk(att_scores, K, largest=True).indices
            neg_idx = torch.topk(att_scores, K, largest=False).indices
            selected_idx = torch.concat((pos_idx, neg_idx), dim=0)

            tiles_labels = torch.zeros(2*K, device=device)
            tiles_labels[:K] = 1
            tiles_probs = tiles_output[selected_idx]
        
        # the loss has two parts, 
        # 1. the slide loss, calculated on the slides classifier output 
        # 2. the tile loss, calculated on the tiles classifier outputs
        tile_loss = criterion(tiles_probs, tiles_labels)
        slide_loss = criterion(output, label)
        loss = lambda1*slide_loss + lambda2*tile_loss

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_tile_loss += tile_loss.item()
        running_slide_loss += slide_loss.item()
        
        train_truth.append(label.item())
        train_pred.append(output.item())
        
        train_tiles_truth.append(tiles_labels)
        train_tiles_probs.append(tiles_probs)

    train_tile_loss = running_tile_loss / len(train_dl)
    
    train_slide_loss = running_slide_loss / len(train_dl)

    train_loss = running_loss / len(train_dl)
    train_auc = roc_auc_score(train_truth, train_pred)
    
    tiles_truth = torch.concat(train_tiles_truth, axis=0).detach().cpu().numpy()
    tiles_probs = torch.concat(train_tiles_probs, axis=0).detach().cpu().numpy()
    train_tiles_auc = roc_auc_score(tiles_truth, tiles_probs)

    return train_loss, train_tile_loss, train_slide_loss, train_auc, train_tiles_auc

def validate(model, val_dl, criterion, device):
    """Predict with model and compute the metrics on val_dl.

    Args:
        model (nn.Module object): model to validate
        val_dl (DataLoader object): validation dataloader
        criterion (Loss object): loss function (used both for slides and tiles)
        device (str): device to run the calculations.

    Returns:
        val_loss, val_auc: loss and AUC of the model on the validation data
    """
    model.eval()

    running_loss = 0
    val_truth = []
    val_pred = []

    for i, (feat, label, is_annot, tile_annot) in enumerate(tqdm(val_dl)):
        feat, label, is_annot, tile_annot = feat.to(device), label.to(device), is_annot.to(device), tile_annot.to(device)
        output, _, _ = model(feat)
        loss = criterion(output, label)

        running_loss += loss.item()
        val_truth.append(label.item())
        val_pred.append(output.item())

    val_loss = running_loss / len(val_dl)
    val_auc = roc_auc_score(val_truth, val_pred)

    return val_loss, val_auc

def learning_curves(n_epochs, train_losses, tile_losses, slide_losses, train_aucs, val_losses=None, val_aucs=None):
    """From losses and AUCs at various epochs, plot the learning curves.

    Args:
        n_epochs (int)): number of epochs (should be the length of all the following arrays)
        train_losses (List of float): global train losses
             train_loss = lambda1*slide_loss + lambda2*tile_loss
        tile_losses (List of float): tile losses
        slide_losses (List of float): slide losses
        train_aucs (List of float): AUCs on the train set
        val_losses (List of float, optional): (slide) validation losses. Defaults to None.
        val_aucs (List of float, optional): validation AUCs. Defaults to None.
    """
    x_axis = range(1,n_epochs+1)
    plt.plot(x_axis, train_losses, 'r', label='train loss')

    if val_losses is not None:
        plt.plot(x_axis, val_losses, 'g', label='val loss')

    plt.plot(x_axis, tile_losses, 'b', label='tile train loss')
    plt.plot(x_axis, slide_losses, 'black', label='slide train loss')
    plt.legend(loc='best')
    plt.show()
    
    plt.plot(x_axis, train_aucs, 'r', label='train auc')

    if val_aucs is not None:
        plt.plot(x_axis, val_aucs, 'g', label='val auc')

    plt.show()

def train(model, train_set, val_set=None, n_epochs=100, reset=True, lambda1=0.5, lambda2=0.5, lr=1e-3, verbose=False, per=0.20, step_size=10, save=False, model_save_path='model.pth', save_last=False, **kwargs):
    """The train general function.

    Args:
        model (nn.Module object): The model to train.
        train_set (Dataset object): The train dataset.
        val_set (_type_, optional): The validation dataset. Defaults to None. If None, the function does not performs validation between epochs.
        n_epochs (int, optional): Number of training epochs. Defaults to 100.
        reset (bool, optional): Whether to reset model parameters (or reinstanciate the object) before training. Defaults to True.
        lambda1 (float, optional): In [0,1], weight for the slide loss. Defaults to 0.5.
        lambda2 (float, optional): In [0,1], weight for the tile loss. Defaults to 0.5.
        lr (_type_, optional): Learning rate. Defaults to 1e-3.
        verbose (bool, optional): Whether to print additional information about the dataset before training. Defaults to False.
        per (float): percentage of pseudo-labels to produce from attention scores 
                    (if n non-annoted tiles, then K=[per*n] positive and K negative pseudo labels are
                     produced by keeping the tiles that have the top-K largest and smallest attention scores). Default to 0.20.
        step_size (int, optional): Number of epochs before decreasing the learning rate. Defaults to 10.
        model_save_path (str, optional): Path where to save the best or last model. Defaults to 'model.pth'.
        save_last (bool, optional): If True, saves the last model. If False, selects the best model across epochs and saves it. Defaults to False.

    Returns:
        best_auc, last_auc, best_epoch (float, float, int) : the best auc reached during the training, the last epoch auc and the best epoch
    """
    # Look if a val dataset is provided. If not, only training will be executed.
    has_val = (val_set is not None)

    if verbose:
        print('Some statistics about the sets')
        compute_dataset_stats(train_set, name='train')
        if has_val:
            compute_dataset_stats(val_set, name='val')
    
    # Before training, if asked to, reinitialize the parameters of the model
    if reset:
        model.reset_parameters()
    
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Batch size can only be set to 1 in this framework (1 slide/~1000 tiles at a time)
    train_dl = DataLoader(train_set, batch_size=1, shuffle=True)
    if has_val:
        val_dl = DataLoader(val_set, batch_size=1, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.6)

    train_losses = []
    train_aucs = []
    tile_losses = []
    slide_losses = []

    if has_val:
        val_losses = []
        val_aucs = []
    
    best_auc = 0
    best_epoch = 0

    for epoch in range(n_epochs):
        train_loss, train_tile_loss, train_slide_loss, train_auc, train_tiles_auc = train_an_epoch(model, train_dl, criterion, optimizer, per, lambda1, lambda2, device)

        slide_losses.append(train_slide_loss)
        tile_losses.append(train_tile_loss)
        train_losses.append(train_loss)
        train_aucs.append(train_auc)

        if has_val:
            val_loss, val_auc = validate(model, val_dl, criterion, device)

            print('Epoch {}: train loss {:.3f} train auc {:.3f} tiles auc {:.3f} val loss {:.3f} val auc {:.3f}'.format(epoch+1, train_loss, train_auc, train_tiles_auc, val_loss, val_auc))
            
            if best_auc < val_auc:
                best_auc = val_auc
                best_epoch = epoch+1
                if save:
                    torch.save(model, model_save_path)

            val_losses.append(val_loss)
            val_aucs.append(val_auc)
            
        else:
            if best_auc < train_auc:
                best_auc = train_auc
                best_epoch = epoch+1
                if save:
                    torch.save(model, model_save_path)

            print('Epoch {}: train loss {:.3f} train auc {:.3f} tiles auc {:.3f}'.format(epoch+1, train_loss, train_auc, train_tiles_auc))

        scheduler.step()
    
    last_auc = (val_aucs[-1] if has_val else train_aucs[-1])

    print('Best auc: {:.3f}, last auc: {:.3f}, best epoch: {}'.format(best_auc, last_auc, best_epoch))
    
    if save and save_last:
        torch.save(model, model_save_path)
    
    # Plot learning curves
    if has_val:
        learning_curves(n_epochs, train_losses, tile_losses, slide_losses, train_aucs, val_losses, val_aucs)
    else:
        learning_curves(n_epochs, train_losses, tile_losses, slide_losses, train_aucs)

    return best_auc, last_auc, best_epoch

def make_cross_val(model, folds, **kwargs):
    """Given a model and a split in N folds of the train dataset, makes a cross-validation in order to 
    make model selection. Hyperparameters can be passed through **kwargs.

    Args:
        model (nn.Model): model 
        folds (List of (str, Dataset) dicts): The folding of the train dataset. 
            For each f in folds, f['train'] gives the train set and f['val'] give the validation set.
    """
    best_aucs = []
    last_aucs = []
    best_epochs = []

    for f in folds:
        best_auc, last_auc, best_epoch = train(model, f['train'], f['val'], save=False, **kwargs)
        best_aucs.append(best_auc)
        last_aucs.append(last_auc)
        best_epochs.append(best_epoch)

    best_aucs = np.array(best_aucs)
    last_aucs = np.array(last_aucs)

    print('Best validation auc averaged over {} folds: {:.3f} (+- {:.3f})'.format(len(folds), best_aucs.mean(), best_aucs.std()))
    print('Last validation auc averaged over {} folds: {:.3f} (+- {:.3f})'.format(len(folds), last_aucs.mean(), last_aucs.std()))
    print('Best epochs: ', best_epochs)

def train_N_models(model, train_dataset, N=10, output_dir='saved_models', **kwargs):
    """Function that trains N models on train_dataset and save them in output_dir.
    Hyperparameters are given through **kwargs.

    Args:
        model (nn.Module object): PyTorch model to train. Will be re-instanciated at each call of the train function (reset=True).
        train_dataset (Dataset object): PyTorch dataset.
        N (int, optional): Number of models. Defaults to 10.
        output_dir (str, optional): path to the directory where the models must be saved. Defaults to 'saved_models'.
    """
    for i in range(N):
        print('Training model n°{}'.format(i))
        train(model, train_dataset, None, save=True, model_save_path=os.path.join(output_dir, 'model_{}.pth'.format(i)), save_last=True, **kwargs)
    
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

def show_attention_maps(model_paths, test_set, w=None):
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