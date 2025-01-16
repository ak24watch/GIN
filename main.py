import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
from GinModel import GINModel
import plotly.graph_objects as go


def split_fold10(labels, fold_idx=0):
    """
    Split the dataset into 10 folds and return the indices for the training and validation sets.

    Args:
        labels (list): List of labels for the dataset.
        fold_idx (int): Index of the fold to use for validation.

    Returns:
        tuple: Indices for the training and validation sets.
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, valid_idx = idx_list[fold_idx]
    return train_idx, valid_idx


def evaluate(dataloader, device, model):
    """
    Evaluate the model on the given dataloader.

    Args:
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run the model on.
        model (nn.Module): Model to evaluate.

    Returns:
        float: Accuracy of the model on the dataset.
    """
    model.eval()
    total = 0
    total_correct = 0
    for batched_graph, labels in dataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        feat = batched_graph.ndata.pop("attr")
        total += len(labels)
        logits = model(batched_graph, feat)
        _, predicted = torch.max(logits, 1)
        total_correct += (predicted == labels).sum().item()
    acc = 1.0 * total_correct / total
    return acc


def plot_accuracy(
    train_acc_list, val_acc_list, dataset_name, filename_prefix="accuracy_plot"
):
    """
    Plot the training and validation accuracy and save the plot as an image file.

    Args:
        train_acc_list (list): List of training accuracies.
        val_acc_list (list): List of validation accuracies.
        dataset_name (str): Name of the dataset.
        filename_prefix (str): Prefix for the filename to save the plot.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(train_acc_list))),
            y=train_acc_list,
            mode="lines+markers",
            name="Train Accuracy",
            line=dict(color="blue"),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(val_acc_list))),
            y=val_acc_list,
            mode="lines+markers",
            name="Validation Accuracy",
            line=dict(color="red"),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        title=f"Training and Validation Accuracy for {dataset_name}",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1]),  # Set y-axis range from 0 to 1
        template="plotly_dark",
        legend=dict(x=0.98, y=0.02, bgcolor='rgba(0,0,0,0)'),
        font=dict(size=14),
    )
    filename = f"{filename_prefix}_{dataset_name}.png"
    fig.write_image(filename)


def plot_best_accuracy(best_acc_dict, filename_prefix="best_accuracy_plot"):
    """
    Plot the best validation accuracy for each dataset and save the plot as an image file.

    Args:
        best_acc_dict (dict): Dictionary with dataset names as keys and best validation accuracies as values.
        filename_prefix (str): Prefix for the filename to save the plot.
    """
    fig = go.Figure()
    datasets = list(best_acc_dict.keys())
    best_accs = list(best_acc_dict.values())
    fig.add_trace(go.Bar(x=datasets, y=best_accs, marker_color="indianred"))
    fig.update_layout(
        title="Best Validation Accuracy Across Datasets",
        xaxis_title="Dataset",
        yaxis_title="Best Validation Accuracy",
        yaxis=dict(range=[0, 1]),  # Set y-axis range from 0 to 1
        template="plotly_dark",
        legend=dict(x=0.98, y=0.02, bgcolor='rgba(0,0,0,0)'),
        font=dict(size=14),
    )
    filename = f"{filename_prefix}.png"
    fig.write_image(filename)


def train(train_loader, val_loader, device, model):
    """
    Train the model on the training set and evaluate on the validation set.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to run the model on.
        model (nn.Module): Model to train.
    """
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    train_acc_list = []
    val_acc_list = []
    for epoch in range(250):
        model.train()
        total_loss = 0
        for batch, (batched_graph, labels) in enumerate(train_loader):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            feat = batched_graph.ndata.pop("attr")
            logits = model(batched_graph, feat)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        train_acc = evaluate(train_loader, device, model)
        valid_acc = evaluate(val_loader, device, model)
        train_acc_list.append(train_acc)
        val_acc_list.append(valid_acc)
        print(
            "Epoch {:05d} | Loss {:.4f} | Train Acc. {:.4f} | Validation Acc. {:.4f} ".format(
                epoch, total_loss / (batch + 1), train_acc, valid_acc
            )
        )
    return train_acc_list, val_acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="MUTAG",
        choices=[
            "MUTAG",
            "PTC",
            "NCI1",
            "PROTEINS",
            "COLLAB",
            "IMDBBINARY",
            "IMDBMULTI",
        ],
        help="name of dataset (default: MUTAG)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="whether to plot the training and validation accuracy",
    )
    args = parser.parse_args()
    print("Training with GIN with a learnable epsilon")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [
        "MUTAG",
        "PTC",
        "NCI1",
        "PROTEINS",
        "COLLAB",
        "IMDBBINARY",
        "IMDBMULTI",
    ]
    if args.plot:
        best_acc_dict = {}
        for dataset_name in datasets:
            dataset = GINDataset(dataset_name, self_loop=False, degree_as_nlabel=True)
            labels = [label for _, label in dataset]
            train_idx, val_idx = split_fold10(labels)
            train_loader = GraphDataLoader(
                dataset,
                sampler=SubsetRandomSampler(train_idx),
                batch_size=128,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = GraphDataLoader(
                dataset,
                sampler=SubsetRandomSampler(val_idx),
                batch_size=128,
                pin_memory=torch.cuda.is_available(),
            )
            in_size = dataset.dim_nfeats
            out_size = dataset.gclasses
            model = GINModel(
                in_size, 16, out_size, num_gin_layers=4, num_mlp_layers=2, dropout=0.05
            ).to(device)
            print(f"Training on {dataset_name}...")
            train_acc_list, val_acc_list = train(
                train_loader, val_loader, device, model
            )
            plot_accuracy(train_acc_list, val_acc_list, dataset_name)
            best_acc_dict[dataset_name] = max(val_acc_list)
        plot_best_accuracy(best_acc_dict)
    else:
        dataset = GINDataset(args.dataset, self_loop=False, degree_as_nlabel=True)
        labels = [label for _, label in dataset]
        train_idx, val_idx = split_fold10(labels)
        train_loader = GraphDataLoader(
            dataset,
            sampler=SubsetRandomSampler(train_idx),
            batch_size=128,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = GraphDataLoader(
            dataset,
            sampler=SubsetRandomSampler(val_idx),
            batch_size=128,
            pin_memory=torch.cuda.is_available(),
        )
        in_size = dataset.dim_nfeats
        out_size = dataset.gclasses
        model = GINModel(
            in_size, 16, out_size, num_gin_layers=4, num_mlp_layers=2, dropout=0.05
        ).to(device)
        print("Training...")
        train(train_loader, val_loader, device, model)
