import numpy as np


def MSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse


def RMSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def MAPE(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def RMSE_MAE_MAPE(y_true, y_pred):
    return (
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        MAPE(y_true, y_pred),
    )
    
    
def RMSE_MAE_MAPE_under_diff_states(y_true, y_pred):
    # (test_size, 12, 207)
    time_step, num_nodes = y_pred.shape
    real = y_true
    pred = y_pred
    lt20_pred = []
    lt20_real = []
    f20_30_pred = []
    f20_30_real = []
    f30_40_pred = []
    f30_40_real = []
    f40_50_pred = []
    f40_50_real = []
    f50_60_pred = []
    f50_60_real = []
    gt60_pred = []
    gt60_real = []
    for i in range(time_step):
        for j in range(num_nodes):
            if real[i][j] < 20:
                lt20_pred.append(pred[i][j])
                lt20_real.append(real[i][j])
            elif 20 <= real[i][j] < 30:
                f20_30_pred.append(pred[i][j])
                f20_30_real.append(real[i][j])
            elif 30 <= real[i][j] < 40:
                f30_40_pred.append(pred[i][j])
                f30_40_real.append(real[i][j])
            elif 40 <= real[i][j] < 50:
                f40_50_pred.append(pred[i][j])
                f40_50_real.append(real[i][j])
            elif 50 <= real[i][j] < 60:
                f50_60_pred.append(pred[i][j])
                f50_60_real.append(real[i][j])
            else:
                gt60_pred.append(pred[i][j])
                gt60_real.append(real[i][j])
    lt20_pred = np.array(lt20_pred)
    lt20_real = np.array(lt20_real)
    f20_30_pred = np.array(f20_30_pred)
    f20_30_real = np.array(f20_30_real)
    f30_40_pred = np.array(f30_40_pred)
    f30_40_real = np.array(f30_40_real)
    f40_50_pred = np.array(f40_50_pred)
    f40_50_real = np.array(f40_50_real)
    f50_60_pred = np.array(f50_60_pred)
    f50_60_real = np.array(f50_60_real)
    gt60_pred = np.array(gt60_pred)
    gt60_real = np.array(gt60_real)
    
    return [[RMSE(lt20_real, lt20_pred), MAE(lt20_real, lt20_pred), MAPE(lt20_real, lt20_pred)],
    [RMSE(f20_30_real, f20_30_pred), MAE(f20_30_real, f20_30_pred), MAPE(f20_30_real, f20_30_pred)],
    [RMSE(f30_40_real, f30_40_pred), MAE(f30_40_real, f30_40_pred), MAPE(f30_40_real, f30_40_pred)],
    [RMSE(f40_50_real, f40_50_pred), MAE(f40_50_real, f40_50_pred), MAPE(f40_50_real, f40_50_pred)],
    [RMSE(f50_60_real, f50_60_pred), MAE(f50_60_real, f50_60_pred), MAPE(f50_60_real, f50_60_pred)],
    [RMSE(gt60_real, gt60_pred), MAE(gt60_real, gt60_pred), MAPE(gt60_real, gt60_pred)]]


def MAE_std(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.std(mae)
        return mae


def std_MAE(y_true, y_pred):
    return MAE_std(y_true, y_pred)


def MSE_RMSE_MAE_MAPE(y_true, y_pred):
    return (
        MSE(y_true, y_pred),
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        MAPE(y_true, y_pred),
    )
