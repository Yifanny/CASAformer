import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
# from torchinfo import summary
import yaml
import json
import sys
import copy
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import normalize, minmax_scale

sys.path.append("..")
from lib.utils import (
    MaskedMAELoss,
    WeightedMaskedMAELoss,
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import RMSE_MAE_MAPE, std_MAE, RMSE_MAE_MAPE_under_diff_states
from lib.my_data_prepare import get_dataloaders_from_index_data, read_distance_matrix
from model.mySTAEformer import mySTAEformer

# ! X shape: (B, T, N, C)


@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch, distance_matrix, SCALER)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    x = []
    y = []
    out = []
    attn_scores = []
    
    i = 0
    for x_batch, y_batch in loader:
        print(i, x_batch.size(), y_batch.size()) 
        # (16, 12, 207, 3) (16, 12, 207, 1)
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch, distance_matrix, SCALER)
        
        # attn_score = torch.sum(attn_score, 1)

        out_batch = SCALER.inverse_transform(out_batch)
        x_batch = SCALER.inverse_transform(x_batch[:, :, :, 0])

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        x_batch = x_batch.cpu().numpy()
        # attn_score = attn_score.cpu().numpy()
        
        # attn_scores.append(attn_score)
        out.append(out_batch)
        y.append(y_batch)
        x.append(x_batch)
        i += 1
        

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()
    x = np.vstack(x).squeeze()
    # attn_scores = np.vstack(attn_scores).squeeze()
   
    return y, out, x #, attn_scores


def train_one_epoch(
    epoch, model, trainset_loader, distance_matrix, optimizer, scheduler, criterion, clip_grad, log=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    i = 0
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        out_batch = model(x_batch, distance_matrix, SCALER)
        
        out_batch = SCALER.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        if loss == 0:
            # print(out_batch)
            exit()
        if i % 100 == 0:
            print(epoch, i, loss)
        i += 1
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss


def train(
    model,
    trainset_loader,
    valset_loader,
    distance_matrix,
    optimizer,
    scheduler,
    train_criterion,
    criterion,
    clip_grad=0,
    max_epochs=200,
    early_stop=10,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    model = model.to(DEVICE)

    wait = 0
    weighted_wait = 0
    min_val_loss = np.inf
    min_weighted_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        print(epoch, wait, weighted_wait)
        train_loss = train_one_epoch(
            epoch, model, trainset_loader, distance_matrix, optimizer, scheduler, train_criterion, clip_grad, log=log
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion)
        weighted_val_loss = eval_model(model, valset_loader, train_criterion)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                "\tTrain Loss = %.5f" % train_loss,
                "Weighted Val Loss = %.5f" % weighted_val_loss,
                "Original Val Loss = %.5f" % val_loss,
                log=log,
            )

        if weighted_val_loss < min_weighted_val_loss:
            weighted_wait = 0
            min_weighted_val_loss = weighted_val_loss
            best_epoch = epoch
            best_state_dict_weight = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict_weight, save+"weighted")
        else:
            weighted_wait += 1
            if wait >= early_stop and weighted_wait >= early_stop:
                break
            
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, save)
        else:
            wait += 1
            if wait >= early_stop and weighted_wait >= early_stop:
                break

    model.load_state_dict(best_state_dict_weight)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
    return model


@torch.no_grad()
def test_model(model, testset_loader, log=None, phase="all"):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred, x_hist = predict(model, testset_loader)
    end = time.time()
    
    print(y_true.shape, y_pred.shape)
    out_steps = y_pred.shape[1]
    all_metrics = []
    for i in range(out_steps):
        metric = RMSE_MAE_MAPE_under_diff_states(y_true[:, i, :], y_pred[:, i, :])
        metric = np.array(metric)
        print(metric.shape)
        all_metrics.append(metric)
    
    print(np.array(all_metrics))
    
    exit()
    
    """
    sample_size, pre_len, node_num = y_true.shape
    congest_true = 0
    congest_false = 0
    free_true = 0
    free_false = 0
    congest_nodes = []
    congest_attn_score = []
    for i in range(sample_size):
        for j in range(node_num):
            y = y_true[i, :, j]
            x = x_hist[i, :, j]
            y_hat = y_pred[i, :, j]
            if np.mean(y) < 0.01:
                continue
            if np.mean(y_hat) >= 60: # and np.mean(y) < 60:
                # congest_true += 1
                congest_nodes.append((i, j))
                continue
            elif np.mean(y_hat) >= 60 and np.mean(y) < 60:
                congest_false += 1
            elif np.mean(y_hat) >= 60 and np.mean(y) >= 60:
                free_true += 1
            elif np.mean(y_hat) < 60 and np.mean(y) >= 60:
                free_false += 1
    print(congest_true, congest_false, free_true, free_false)
    distance_matrix = read_distance_matrix("../data/METRLA")
    
    distance_nodes_congested_high_attn_scores = []
    nodes_congested_high_attn_scores = []
    for i, j in congest_nodes:
        t = i
        node_id = j
        attn_matrix = attn_scores[t * 4 : t * 4 + 4]
        congested_attn_scores = np.sum(attn_matrix, 0)[j]
        srt_idx = np.argsort(congested_attn_scores)
        srt_idx = np.delete(srt_idx, np.where(srt_idx == j))
        
        distance_nodes_congested_high_attn_scores.append(distance_matrix[j][srt_idx[-3::]])
        nodes_congested_high_attn_scores.append(x_hist[i, :, srt_idx[-3::]])
        
    distance_nodes_congested_high_attn_scores = np.hstack(distance_nodes_congested_high_attn_scores).squeeze() 
    nodes_congested_high_attn_scores = np.vstack(nodes_congested_high_attn_scores).squeeze()   
    nodes_congested_high_attn_scores = np.mean(nodes_congested_high_attn_scores, axis=-1)
    print(distance_nodes_congested_high_attn_scores.shape, nodes_congested_high_attn_scores.shape)
    H, xedges, yedges = np.histogram2d(nodes_congested_high_attn_scores, distance_nodes_congested_high_attn_scores)
    print(H.shape)
    print(H)
    print(xedges, yedges)
    plt.figure()
    plt.imshow(H.T, interpolation='nearest', origin='lower',
        extent=[xedges[0] * 300, xedges[-1] * 300, yedges[0], yedges[-1]])
    plt.xticks((xedges * 300).astype(int), xedges.astype(int))
    plt.xlabel("avg historical speed")
    plt.ylabel("distance")
    plt.savefig("distance_avg_speed_gt60_high_attn_scores.png")
    exit()
    plt.figure()
    plt.hist(distance_nodes_congested_high_attn_scores)
    # plt.scatter(distance_nodes_congested_high_attn_scores, nodes_congested_high_attn_scores)
    # plt.xlabel("distance")
    # plt.ylabel("avg speed")
    plt.savefig("distance_lt50_high_attn_scores.png")
    exit()
    """

    # y_true = y_true.flatten()
    # print(y_true.shape)

    # plt.figure()
    # plt.hist(y_true, bins=50)
    # plt.savefig("speed.png")

    # print(y_true.shape, y_pred.shape)
    """
    if phase == "free":
        sample_size, pre_len, node_num = y_true.shape
        new_y_true = []
        new_y_pred = []
        for i in range(sample_size):
            for j in range(node_num):
                y = y_true[i, :, j]
                if np.mean(y) >= 60:
                    new_y_true.append(y)
                    new_y_pred.append(y_pred[i, :, j])
        new_y_true = np.array(new_y_true)
        new_y_pred = np.array(new_y_pred)
        new_y_true = np.expand_dims(new_y_true, -1)
        new_y_pred = np.expand_dims(new_y_pred, -1)
        y_true = new_y_true
        y_pred = new_y_pred
    elif phase == "congested":
        sample_size, pre_len, node_num = y_true.shape
        new_y_true = []
        new_y_pred = []
        for i in range(sample_size):
            for j in range(node_num):
                y = y_true[i, :, j]
                if np.mean(y) < 60:
                    new_y_true.append(y)
                    new_y_pred.append(y_pred[i, :, j])
        new_y_true = np.array(new_y_true)
        new_y_pred = np.array(new_y_pred)
        new_y_true = np.expand_dims(new_y_true, -1)
        new_y_pred = np.expand_dims(new_y_pred, -1)
        y_true = new_y_true
        y_pred = new_y_pred

    # estimate the change of the state
    f2f_acc = 0
    f2s_acc = 0
    s2s_acc = 0
    s2f_acc = 0
    all_f2f = 0
    all_f2s = 0
    all_s2s = 0
    all_s2f = 0
    sample_size, pre_len, node_num = y_true.shape
    for i in range(sample_size):
        for j in range(node_num):
            y = y_true[i, :, j]
            x = x_hist[i, :, j]
            y_hat = y_pred[i, :, j]
            if np.mean(x) >= 60 and np.mean(y) >= 60:
                all_f2f += 1
                if np.mean(y_hat) >= 60:
                    f2f_acc += 1
            elif np.mean(x) >= 60 and np.mean(y) < 60:
                all_f2s += 1
                if np.mean(y_hat) < 60:
                    f2s_acc += 1
            elif np.mean(x) < 60 and np.mean(y) < 60:
                all_s2s += 1
                if np.mean(y_hat) < 60:
                    s2s_acc += 1   
            elif np.mean(x) < 60 and np.mean(y) >= 60:
                all_s2f += 1
                if np.mean(y_hat) >= 60:
                    s2f_acc += 1    
    f2f_acc = f2f_acc / all_f2f
    f2s_acc = f2s_acc / all_f2s
    s2s_acc = s2s_acc / all_s2s
    s2f_acc = s2f_acc / all_s2f

    print("free to free", f2f_acc)
    print("free to congest", f2s_acc)
    print("congest to congest", s2s_acc)
    print("congest to free", s2f_acc)
    """

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        mae_std = std_MAE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f, MAE std = %.5f \n" % (
            i + 1,
            rmse,
            mae,
            mape,
            mae_std
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="PEMSBAY")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    args = parser.parse_args()

    seed = torch.randint(1000, (1,)) # set random seed here
    seed_everything(seed)
    set_cpu_num(1)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("device:", DEVICE)

    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"
    model_name = mySTAEformer.__name__

    with open(f"{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # -------------------------------- load model -------------------------------- #

    model = mySTAEformer(**cfg["model_args"])

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"my-sparse_attn_-{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #
    
    print_log(dataset, log=log)
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
    ) = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        log=log,
    )
    print_log(log=log)
    
    distance_matrix = read_distance_matrix(f"../data/{dataset}")
    distance_matrix = 1 / distance_matrix
    distance_matrix[np.isinf(distance_matrix)] = 1
    distance_matrix = minmax_scale(distance_matrix, axis=0)
    distance_matrix = torch.tensor(distance_matrix).to(torch.float32).to(DEVICE)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-sparse_attn-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    if dataset in ("METRLA", "PEMSBAY"):
        train_criterion = WeightedMaskedMAELoss()
        criterion = MaskedMAELoss()
    elif dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        criterion = nn.HuberLoss()
    else:
        raise ValueError("Unsupported dataset.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    # print_log(
    #     summary(
    #         model,
    #         [
    #             cfg["batch_size"],
    #             cfg["in_steps"],
    #             cfg["num_nodes"],
    #             next(iter(trainset_loader))[0].shape[-1],
    #         ],
     #        verbose=0,  # avoid print twice
     #    ),
     #    log=log,
    # )
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)
    
    # model.load_state_dict(torch.load("../saved_models/mySTAEformer-with-congest-mask-METRLA-2024-07-05-16-19-24.pt", map_location=DEVICE))
    model = train(
        model,
        trainset_loader,
        valset_loader,
        distance_matrix,
        optimizer,
        scheduler,
        train_criterion,
        criterion,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        verbose=1,
        log=log,
        save=save,
    )

    print_log(f"Saved Model: {save}", log=log)
    
    model.to(DEVICE)

    test_model(model, testset_loader, log=log)

    log.close()
