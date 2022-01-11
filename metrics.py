import numpy as np

def MAE(pred, y, axis = 0):
    return np.nanmean(np.abs(pred - y), axis = axis)

def RMSE(pred, y, axis = 0):
    return np.sqrt(np.nanmean((pred - y)**2, axis = axis))

def SMAPE(pred, y, axis = 0):
    return 200 * np.nanmean(np.abs(pred - y) / (np.abs(pred) + np.abs(y)), axis = axis)

def R(pred, y, axis = 0):
    mask = ~np.isnan(y)
    n = mask.sum(axis = axis, keepdims = True)
    pred_mean = np.sum(pred * mask, axis = axis, keepdims = True) / n
    y_mean = np.nanmean(y, axis = axis, keepdims = True)
    r = np.nansum((pred - pred_mean) * (y - y_mean), axis = axis)/ np.sqrt(np.sum(mask * (pred - pred_mean)**2, axis = axis) * np.nansum((y - y_mean)**2, axis = axis))
    return r

metrics = [MAE, RMSE, SMAPE, R]