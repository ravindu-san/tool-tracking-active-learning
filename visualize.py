import numpy as np
import matplotlib.pyplot as plt
import torch

# from data import overlap
from lstm import create_windowed_ts, stratified_sampling

def visualize_class_counts_bar(y, title=None, path=None):
    values, counts = np.unique(y, return_counts=True)

    # Create positions 0, 1, 2, ..., number_of_classes-1
    x = np.arange(len(values))

    plt.figure(figsize=(14, 8))
    bars = plt.bar(x, counts, color='skyblue', edgecolor='black')

    # Add count labels above each bar
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha='center', va='bottom', fontsize=12
        )

    plt.xlabel('Class Label')
    plt.ylabel('Sample (Window) Count')
    plt.title(title)

    # Use the class labels as ticks
    plt.xticks(x, values)

    if path is not None:
        plt.savefig(path, bbox_inches='tight')

    plt.show()



if __name__ == '__main__':
    Xt_acc, Xt_gyr, Xt_mag, Xt_mic, y_org_label = create_windowed_ts(data_path="./tool-tracking-data/",
                                                          tool="electric_screwdriver",
                                                                     window_length=0.4,
                                                                     overlap=0.25, invalid_classes=[])

    visualize_class_counts_bar(y_org_label, title="Class Distribution in Filtered out Whole Dataset", path='./visualizations/whole_class_counts_bar.png')

    # train_dataset, val_dataset, test_dataset = stratified_sampling(torch.from_numpy(Xt_acc),
    #                                                                torch.from_numpy(Xt_gyr),
    #                                                                torch.from_numpy(Xt_mag),
    #                                                                torch.from_numpy(Xt_mic),
    #                                                                torch.tensor(y_org_label))

    # y_train = train_dataset.dataset[train_dataset.indices][-1]
    # y_val= val_dataset.dataset[val_dataset.indices][-1]
    # y_test= test_dataset.dataset[test_dataset.indices][-1]
    #
    # visualize_class_counts_bar(y_train, title="Class Distribution in Train Dataset", path='./visualizations/train_class_counts_bar.png')
    # visualize_class_counts_bar(y_val, title="Class Distribution in Validation Dataset", path='./visualizations/val_class_counts_bar.png')
    # visualize_class_counts_bar(y_test, title="Class Distribution in Test Dataset", path='./visualizations/test_class_counts_bar.png')
