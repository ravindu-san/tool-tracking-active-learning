from typing import List

import numpy as np
from datatools import MeasurementDataReader, Tool, Config, MeasurementSeries, Measurement, DataTypes, Action
from datatools import ACC, GYR, MAG, MIC, POS, VEL
from datatools import to_ts_data
from fhgutils import Segment, contextual_recarray_dtype, filter_ts_data
import numpy as np
from seglearn.base import TS_Data
from seglearn.pipe import Pype
from fhgutils import filter_labels, one_label_per_window, summarize_labels


def has_unique_label_for_window(y):
    values, counts = np.unique(y, return_counts=True)
    # idx = np.argmax(counts)
    max_count = np.max(counts)
    if max_count > 0.5 * np.sum(counts):
        return True
    else: return False

def create_windowed_ts(data_path: str, tool: str, invalid_classes: List[int], window_length: float = 0.4, overlap: float = 0.25):
    mdr = MeasurementDataReader(source=data_path)
    data_dict = mdr.query().filter_by(Tool == tool).get()

    Xt, Xc, y = to_ts_data(data_dict, contextual_recarray_dtype)
    X = TS_Data(Xt, Xc)

    pipe = Pype([
        ('segment', Segment(window_length=window_length, overlap=overlap, enforce_size=True, n=len(np.unique(Xc.desc))))
    ])

    X_trans, y_trans = pipe.fit_transform(X, y)

    Xt_acc, Xc_acc, y_acc = filter_ts_data(X_trans, y_trans, filt={'desc': ['acc']})
    Xt_gyr, Xc_gyr, y_gyr = filter_ts_data(X_trans, y_trans, filt={'desc': ['gyr']})
    Xt_mag, Xc_mag, y_mag = filter_ts_data(X_trans, y_trans, filt={'desc': ['mag']})
    Xt_mic, Xc_mic, y_mic = filter_ts_data(X_trans, y_trans, filt={'desc': ['mic']})

    keep_mask = np.ones(len(y_acc), dtype=bool)
    y = []

    count_1 = 0
    count_inval_classes = 0
    count_maj_vote = 0
    count_diff_label = 0

    for i in range(len(y_acc)):
        # remove windows with label -1
        if (-1 in y_acc[i]) or (-1 in y_gyr[i]) or (-1 in y_mag[i]) or (-1 in y_mic[i]):
            # print("Warning: skipping window {}".format(i))
            keep_mask[i] = False
            count_1 += 1
            continue

        # if (any(v in y_acc[i] for v in invalid_classes) or
        #         any(v in y_gyr[i] for v in invalid_classes) or
        #         any(v in y_mag[i] for v in invalid_classes) or
        #         any(v in y_mic[i] for v in invalid_classes)):
        #     print(f"Warning: skipping window {i}")
        #     keep_mask[i] = False
        #     count_1_14 += 1
        #     continue

        # remove windows that doesn't have majority label
        if (not has_unique_label_for_window(y_acc[i])) or (not has_unique_label_for_window(y_gyr[i])) or (not has_unique_label_for_window(y_mag[i])) or (not has_unique_label_for_window(y_mic[i])):
            # print("Warning: skipping window {}".format(i))
            keep_mask[i] = False
            count_maj_vote += 1
            continue

        val_acc, counts_acc = np.unique(y_acc[i], return_counts=True)
        idx = np.argmax(counts_acc)
        y_acc_window = int(val_acc[idx])

        val_gyr, counts_gyr = np.unique(y_gyr[i], return_counts=True)
        idx = np.argmax(counts_gyr)
        y_gyr_window = int(val_gyr[idx])

        val_mag, counts_mag = np.unique(y_mag[i], return_counts=True)
        idx = np.argmax(counts_mag)
        y_mag_window = int(val_mag[idx])

        val_mic, counts_mic = np.unique(y_mic[i], return_counts=True)
        idx = np.argmax(counts_mic)
        y_mic_window = int(val_mic[idx])

        # remove windows that has different labels across sensor types
        if not (y_acc_window == y_gyr_window == y_mag_window == y_mic_window):
            # print("Warning: skipping window {}".format(i))
            keep_mask[i] = False
            count_diff_label += 1
            continue

        if (y_acc_window in invalid_classes) or (y_gyr_window in invalid_classes) or (y_mag_window in invalid_classes) or (y_mic_window in invalid_classes):
            # print("Warning: skipping window {}".format(i))
            keep_mask[i] = False
            count_inval_classes += 1
            continue

        assert y_acc_window == y_gyr_window == y_mag_window == y_mic_window
        y.append(y_acc_window)


    val_y, counts_y = np.unique(y, return_counts=True)
    print(val_y, counts_y)

    Xt_acc_f = Xt_acc[keep_mask]
    Xt_gyr_f = Xt_gyr[keep_mask]
    Xt_mag_f = Xt_mag[keep_mask]
    Xt_mic_f = Xt_mic[keep_mask]

    print(f"count_1_14: {count_1}, count_invalid_classes: {count_inval_classes}, count_maj_vote: {count_maj_vote}, count_diff_label: {count_diff_label}")

    assert Xt_acc_f.shape[0] == Xt_gyr_f.shape[0] == Xt_mag_f.shape[0] == Xt_mic_f.shape[0] == len(y)
    # return Xt_acc_f, Xt_gyr_f, Xt_mag_f, Xt_mic_f, y
    return np.stack(Xt_acc_f).astype(np.float64), np.stack(Xt_gyr_f).astype(np.float64), np.stack(Xt_mag_f).astype(np.float64), np.stack(Xt_mic_f).astype(np.float64), y


def map_class_label_to_idx(y: List[int]):
    class_mapping = {
        2: 0,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
        8: 6,
        14: 7
    }

    y_mapped = [class_mapping[label] for label in y]
    return y_mapped
