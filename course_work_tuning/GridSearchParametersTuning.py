import numpy as np
import pandas as pd
import importlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score
import pickle as pkl
import pyeeg
import pyrem as pruni
import scipy.signal as sig
import scipy.fftpack as fft


def recurrence(series, epsilon=1.0):
    return (np.abs(series - series[:, np.newaxis]) < epsilon).sum() / (series.shape[0] * series.shape[0])


def normalized_crosscorrelation(series1, series2):
    if series1.shape[0] != series2.shape[0]:
        raise BaseException('shapes must be the same')
    if (series1.shape[0] == 1) or (series2.shape[0] == 1):
        if np.isclose(series1[0], 0.0, rtol=0.0, atol=0.000001) or np.isclose(series2[0], 0.0, rtol=0.0,
                                                                              atol=0.000001):
            return 0.0
        if np.isclose(series1[0], series2[0], rtol=0.0, atol=0.000001):
            return 1.0
        if np.isclose(abs(series1[0]), abs(series2[0]), rtol=0.0, atol=0.000001):
            return -1.0
        return 0.0
    return ((series1 - series1.mean()) * (series2 - series2.mean())).sum() /\
            (series1.shape[0] * series1.std() * series2.std())


def normalized_autocorrelation(series, max_lag):
    result = []
    result.append(normalized_crosscorrelation(series, series))
    if max_lag <= 0:
        return np.array(result)
    if max_lag >= series.shape[0]:
        raise BaseException('max_lag must be not more than series length')
    temp_series = np.hstack((series, series))
    for k in range(1, max_lag + 1):
        result.append(normalized_crosscorrelation(temp_series[:-k], temp_series[k:]))
    return np.array(result)


def mean_autocorrelation(series, max_lag):
    return normalized_autocorrelation(series, max_lag).mean()


def mean_period(series, max_lag, threshold=-1.0):
    '''left local minimum and all local maximums must be uniquely defined'''
    if series.shape[0] < 3:
        raise BaseException('time series must be longer')
    auto_corr = normalized_autocorrelation(series, max_lag)
    auto_corr[auto_corr < threshold] = threshold
    diffs = np.diff(auto_corr)
    prod_diffs = diffs[:-1] * diffs[1:]
    susp_extr_dots = np.where(prod_diffs < 0)[0]
    start_index = np.array([0])
    if (diffs[susp_extr_dots[0]] < 0) and (diffs[susp_extr_dots[0] + 1] > 0):
        if susp_extr_dots.shape[0] > 1:
            start_index = np.hstack((0, susp_extr_dots[1::2], series.shape[0]))
    else:
        start_index = np.hstack((0, susp_extr_dots[::2], series.shape[0]))
    if start_index.shape[0] == 1:
        start_index = np.hstack((start_index, series.shape[0]))
    return np.diff(start_index).mean()


def poincare_SD(series):
    scaled = StandardScaler().fit_transform(series.reshape(-1, 1)).ravel()
    pp = np.hstack((scaled[:-1][:, np.newaxis], scaled[1:][:, np.newaxis]))
    p1 = np.array([[-1.0 / np.sqrt(2.0)], [1.0 / np.sqrt(2.0)]])
    p2 = np.array([[1.0 / np.sqrt(2.0)], [1.0 / np.sqrt(2.0)]])
    return np.dot(pp, p1).ravel().std(), np.dot(pp, p2).ravel().std()


def DET_and_mean_diag_length(series, epsilon=1.0, min_l=1):
    if (min_l > series.shape[0]) or (min_l < 1):
        raise BaseException('min_l must be in correct range')
    rp = (np.abs(series - series[:, np.newaxis]) < epsilon).astype(np.int)
    lines_hist = np.zeros(series.shape[0]).astype(np.int)
    isline = False
    length = 0
    for j in range(1, series.shape[0]):
        for k in range(series.shape[0] - j):
            if rp[k][j + k]:
                if isline:
                    length += 1
                else:
                    isline = True
                    length = 1
            else:
                isline = False
                if length:
                    lines_hist[length - 1] += 1
                    length = 0
        isline = False
        if length:
            lines_hist[length - 1] += 1
            length = 0
    lines_hist *= 2
    lines_hist[-1] += 1
    line_lengths = np.arange(series.shape[0]) + 1
    mask = line_lengths >= min_l
    line_lengths[~mask] = 0
    sum_length = (line_lengths * lines_hist).sum()
    return sum_length / rp.sum(), sum_length / lines_hist[mask].sum()


def FilterBank(X, filters='BandpassBank', sample_rate=128, butter_pows=None):
    freq_pairs = None
    X_processed = None
    if filters == 'BandpassBank':
        freq_pairs = [[0.5], [0.5, 4.0], [4.0, 8.0], [8.0, 13.0], [13.0, 30.0], [30.0, 42.0]]
        butter_pows = [5, 6, 8, 11, 11, 10]
    else:
        freq_pairs = filters
    for i in range(len(freq_pairs)):
        power = 5 if butter_pows is None else butter_pows[i]
        if len(freq_pairs[i]) == 1:
            b, a = sig.butter(power, 2 * freq_pairs[i][0] / sample_rate, btype='lowpass')
        else:
            b, a = sig.butter(power, 2 * np.array(freq_pairs[i]) / sample_rate, btype='bandpass')
        X_filtered = sig.lfilter(b, a, X, axis=0)
        X_processed = X_filtered if X_processed is None else np.c_[X_processed, X_filtered]
    return X_processed


def WindowDataset(dataset, interval=77, shift=1):
    if (interval < 1) or (interval > dataset.shape[0]):
        raise BaseException('interval has invalid value')
    if (shift < 1) or (shift > interval):
        raise BaseException('shift has invalid value')
    windowed_dataset = np.zeros((np.ceil((dataset.shape[0] - interval + 1) / shift).astype(np.int),
                                 interval, dataset.shape[1]))
    begin = 0
    for i in range(windowed_dataset.shape[0]):
        windowed_dataset[i] = dataset[begin: begin + interval, :].copy()
        begin += shift
    return windowed_dataset


if __name__ == '__main__':
    print('Start tuning ...')
    dataset_np = pd.read_csv('./eeg_eye_state.csv', sep=',').as_matrix()
    print('Loaded dataset ...')
    windowed_dataset = WindowDataset(dataset_np, interval=77, shift=1)
    target = np.zeros(windowed_dataset.shape[0]).astype(np.int)
    for i in range(target.shape[0]):
        target[i] = windowed_dataset[i][38, 14]
    print('Splited dataset into windows ...')
    train_windowed, test_windowed = windowed_dataset[:7452], windowed_dataset[7452:]
    train_y, test_y = target[:7452], target[7452:]
    print('Train/Test split completed ...')
    scaler = StandardScaler()
    scaler.fit(dataset_np[:7528, :14])
    for i in range(train_windowed.shape[0]):
        train_windowed[i, :, :14] = scaler.transform(train_windowed[i, :, :14])
    for i in range(test_windowed.shape[0]):
        test_windowed[i, :, :14] = scaler.transform(test_windowed[i, :, :14])
    print('Scaled dataset ...')
    train_banded = np.zeros((train_windowed.shape[0], train_windowed.shape[1], 14 * 6))
    test_banded = np.zeros((test_windowed.shape[0], test_windowed.shape[1], 14 * 6))
    for i in range(train_banded.shape[0]):
        train_banded[i] = FilterBank(train_windowed[i, :, :14])
    for i in range(test_banded.shape[0]):
        test_banded[i] = FilterBank(test_windowed[i, :, :14])
    print('Extracted waves ...')
    print('Tuning recurrence ...')
    tolerance_values = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])
    opt_tols = []
    for i in range(train_banded.shape[2]):
        table = []
        for tol in tolerance_values:
            recurrence_list = []
            for j in range(train_banded.shape[0]):
                recurrence_list.append(recurrence(train_banded[j, :, i], epsilon=tol))
            recurrence_list = np.array(recurrence_list)
            score = np.max([roc_auc_score(y_true=train_y, y_score=recurrence_list),
                            roc_auc_score(y_true=1 - train_y, y_score=recurrence_list)])
            table.append([score, tol])
        table = np.array(table)
        idx = np.argmax(table[:, 0])
        opt_tols.append({'channel': i, 'tol': table[idx, 1], 'score': table[idx, 0]})
        print('    Channel %i: optimal value is'%(i), table[idx, 1], 'tol --', 'auc roc is', table[idx, 0])
    print('Tuning determinism and mean diagonal length ...')
    min_l_values = np.array([5, 10, 15, 20, 25])
    opt_min_l_tols = []
    for i in range(train_banded.shape[2]):
        table = []
        for min_l in min_l_values:
            for tol in tolerance_values:
                det_min_l_list = []
                for j in range(train_banded.shape[0]):
                    det_min_l_list.append(list(DET_and_mean_diag_length(train_banded[j, :, i], epsilon=tol, min_l=min_l)))
                det_min_l_list = np.array(det_min_l_list)
                det_score = np.max([roc_auc_score(y_true=train_y, y_score=det_min_l_list[:, 0]),
                                roc_auc_score(y_true=1 - train_y, y_score=det_min_l_list[:, 0])])
                min_l_score = np.max([roc_auc_score(y_true=train_y, y_score=det_min_l_list[:, 1]),
                                roc_auc_score(y_true=1 - train_y, y_score=det_min_l_list[:, 1])])
                score = np.max([det_score, min_l_score])
                table.append([score, min_l, tol])
        table = np.array(table)
        idx = np.argmax(table[:, 0])
        opt_min_l_tols.append({'channel': i, 'min_l': table[idx, 1], 'tol': table[idx, 2], 'score': table[idx, 0]})
        print('    Channel %i: optimal pair is'%(i), table[idx, 1], 'min_l', table[idx, 2], 'tol --', 'auc roc is',
          table[idx, 0])
    print('Tuning aproximate entropy ...')
    subwindow_sizes = np.array([5, 10, 15, 20, 25])
    opt_sw_tol_apent = []
    for i in range(train_banded.shape[2]):
        table = []
        for subwindow in subwindow_sizes:
            for tol in tolerance_values:
                entropy_list = []
                for j in range(train_banded.shape[0]):
                    entropy_list.append(pyeeg.ap_entropy(train_banded[j, :, i], subwindow, tol))
                entropy_list = np.array(entropy_list)
                score = np.max([roc_auc_score(y_true=train_y, y_score=entropy_list),
                                roc_auc_score(y_true=1 - train_y, y_score=entropy_list)])
                table.append([score, subwindow, tol])
        table = np.array(table)
        idx = np.argmax(table[:, 0])
        opt_sw_tol_apent.append({'channel': i, 'subwindow': table[idx, 1], 'tol': table[idx, 2], 'score': table[idx, 0]})
        print('    Channel %i: optimal pair is'%(i), table[idx, 1], 'subwindow', table[idx, 2], 'tol --', 'auc roc is',
              table[idx, 0])
    print('Tuning sample entropy ...')
    opt_sw_tol_sampent = []
    for i in range(train_banded.shape[2]):
        table = []
        for subwindow in subwindow_sizes:
            for tol in tolerance_values:
                entropy_list = []
                for j in range(train_banded.shape[0]):
                    entropy_list.append(pyeeg.samp_entropy(train_banded[j, :, i], subwindow, tol))
                entropy_list = np.array(entropy_list)
                score = np.max([roc_auc_score(y_true=train_y, y_score=entropy_list),
                                roc_auc_score(y_true=1 - train_y, y_score=entropy_list)])
                table.append([score, subwindow, tol])
        table = np.array(table)
        idx = np.argmax(table[:, 0])
        opt_sw_tol_sampent.append({'channel': i, 'subwindow': table[idx, 1], 'tol': table[idx, 2], 'score': table[idx, 0]})
        print('    Channel %i: optimal pair is'%(i), table[idx, 1], 'subwindow', table[idx, 2], 'tol --', 'auc roc is',
              table[idx, 0])
    tuning_results = {'recurrence': opt_tols, 'det_min_l': opt_min_l_tols, 'apent': opt_sw_tol_apent, 'sampent': opt_sw_tol_sampent}
    fileobj = open('result.pkl', 'wb')
    pkl.dump(tuning_results, fileobj)
    fileobj.close()
    
