import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange
import csv
import pickle
# import matplotlib.pyplot as plt

# ! X shape: (B, T, N, C)


def get_dataloaders_from_index_data(
    data_dir, tod=False, dow=False, dom=False, batch_size=64, log=None
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    # if dom:
    #     features.append(3)
    data = data[..., features]

    index = np.load(os.path.join(data_dir, "index.npz"))

    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]
    
    # print(x_train.shape)
    # lt40 = 0
    # b40_50 = 0
    # b50_60 = 0
    # gt60 = 0
    # for i in range(x_train.shape[0]):
    #     for j in range(207):
    #         if np.mean(x_train[i][:, j, :]) < 40:
    #             lt40 += 1
    #         elif 40 <= np.mean(x_train[i][:, j, :]) < 50:
    #             b40_50 += 1
    #         elif 50 <= np.mean(x_train[i][:, j, :]) < 60:
    #             b50_60 += 1
    #         else:
    #             gt60 += 1
    # print(lt40, b40_50, b50_60, gt60) # 822793 283014 925147 2931664

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
    # print(scaler.transform(50))
    # print(scaler.transform(0))
    # exit()

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler


def get_balanced_dataloaders_from_index_data(
    data_dir, tod=False, dow=False, dom=False, batch_size=64, log=None
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    # if dom:
    #     features.append(3)
    data = data[..., features]

    index = np.load(os.path.join(data_dir, "index.npz"))

    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]
    
    """
    y_mean = np.mean(y_train, (1, 2, 3))
    new_x_train = []
    new_y_train = []
    n = [0,0,0,0,0,0]
    for t in range(len(y_mean)):
        if y_mean[t] == 0:
            continue
        if y_mean[t] > 60:
            n[5] += 1
        elif 50 < y_mean[t] <= 60:
            n[4] += 1
        elif 40 < y_mean[t] <= 50:
            n[3] += 1
        elif 30 < y_mean[t] <= 40:
            n[2] += 1
        elif 20 < y_mean[t] <= 30:
            n[1] += 1
        elif y_mean[t] <= 20:
            n[0] += 1
    print(n) # [165, 86, 203, 2758, 10666, 9098]
    """
    
    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler


def get_classification_dataloaders_from_index_data(
    data_dir, tod=False, dow=False, dom=False, batch_size=64, log=None
):
    data = np.load(os.path.join(data_dir, "data.npz"))["data"].astype(np.float32)

    features = [0]
    if tod:
        features.append(1)
    if dow:
        features.append(2)
    # if dom:
    #     features.append(3)
    data = data[..., features]

    index = np.load(os.path.join(data_dir, "index.npz"))

    train_index = index["train"]  # (num_samples, 3)
    val_index = index["val"]
    test_index = index["test"]

    x_train_index = vrange(train_index[:, 0], train_index[:, 1])
    y_train_index = vrange(train_index[:, 1], train_index[:, 2])
    x_val_index = vrange(val_index[:, 0], val_index[:, 1])
    y_val_index = vrange(val_index[:, 1], val_index[:, 2])
    x_test_index = vrange(test_index[:, 0], test_index[:, 1])
    y_test_index = vrange(test_index[:, 1], test_index[:, 2])

    x_train = data[x_train_index]
    y_train = data[y_train_index][..., :1]
    y_train = (np.mean(y_train, 1)/10).astype(int)
    x_val = data[x_val_index]
    y_val = data[y_val_index][..., :1]
    y_val = (np.mean(y_val, 1)/10).astype(int)
    x_test = data[x_test_index]
    y_test = data[y_test_index][..., :1]
    y_test = (np.mean(y_test, 1)/10).astype(int)

    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())

    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler
        

def read_distance_matrix(data_dir):
    with open(os.path.join(data_dir, "adj_mx.pkl"), 'rb') as f:
        adj_mx = pickle.load(f, encoding='latin1')
        sensor_id_dict = adj_mx[1]
    
    distance_matrix = np.ones((len(sensor_id_dict), len(sensor_id_dict))) * 20000
    with open(os.path.join(data_dir, "distances.csv"), 'r') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i == 0:
                i += 1
                continue
            fr = row[0]
            to = row[1]
            d = float(row[2])
            if fr in sensor_id_dict and to in sensor_id_dict:
                distance_matrix[sensor_id_dict[fr]][sensor_id_dict[to]] = d
    
    return distance_matrix

# get_classification_dataloaders_from_index_data("../data/METRLA")
# read_distance_matrix("../data/METRLA")
# get_dataloaders_from_index_data("../data/METRLA")