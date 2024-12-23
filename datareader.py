import numpy as np
import h5py
import pandas as pd
from torch.utils.data import Dataset

class HeartSoundsDataset(Dataset):
    def __init__(self, csv_path, hdf5_path, transform=None):
        """
        初始化数据集，加载CSV文件中的元数据，准备标签。

        参数:
            csv_path (str): 指定临床数据 CSV 文件的路径。
            hdf5_path (str): 指定 HDF5 文件的路径，包含信号数据。
            transform (callable, optional): 用于对信号数据进行变换的函数。
        """
        self.csv_path = csv_path
        self.hdf5_path = hdf5_path
        self.transform = transform

        # 加载CSV文件
        self.metadata = pd.read_csv(csv_path)
        # self.metadata =self.metadata[:2000]
        # 筛选必要的列
        self.paths = self.metadata['path']
        self.labels = self.metadata['Clinical_Diagnosis'].apply(lambda x: 1 if x == 'Normal' else 0)

        # 计算所有信号的最短长度
        self.max_length = self.calculate_min_signal_length()

    def calculate_min_signal_length(self):
        """
        计算所有信号的最短长度。

        返回:
            int: 最短信号的长度。
        """
        min_length = float('inf')  # 初始为无穷大
        for signal_path in self.paths:
            with h5py.File(self.hdf5_path, 'r') as hdf5_file:
                signal = np.array(hdf5_file[signal_path])
                min_length = min(min_length, len(signal))  # 更新最短长度
        return min_length

    def __len__(self):
        """
        返回数据集的大小。
        """
        return len(self.paths)

    def trim_signal(self, signal, max_length):
        """
        裁剪信号数据到指定的最大长度。

        参数:
            signal (numpy.ndarray): 输入信号。
            max_length (int): 目标长度。
        
        返回:
            numpy.ndarray: 裁剪后的信号。
        """
        current_length = len(signal)
        if current_length > max_length:
            # 如果信号长度超过最大长度，裁剪掉多余的部分
            signal = signal[:max_length]
        return signal

    def __getitem__(self, idx):
        """
        获取给定索引的信号和标签。

        参数:
            idx (int): 需要获取的样本的索引。

        返回:
            tuple: (信号, 标签)，其中信号是波形数据，标签是 0 或 1。
        """
        signal_path = self.paths.iloc[idx]
        label = self.labels.iloc[idx]

        # 从HDF5文件中读取信号
        with h5py.File(self.hdf5_path, 'r') as hdf5_file:
            signal = np.array(hdf5_file[signal_path])

        # 使用最短信号长度裁剪信号
        signal = self.trim_signal(signal, self.max_length)
        signal = signal.flatten()
        signal = (signal - signal.mean()) / signal.std()
        # 如果有变换函数，应用变换
        if self.transform:
            signal = self.transform(signal)

        return signal, label


if __name__ == "__main__":
    # Define file paths
    csv_file_path = "D:\\sjq\\heart_sounds\\code\\data\\clinical_data_expanded.csv"
    hdf5_file_path = "D:\\sjq\\heart_sounds\\Clinical Trials\\clinical_study_2024_dataset.hdf5"

    # Initialize the dataset
    dataset = HeartSoundsDataset(csv_path=csv_file_path, hdf5_path=hdf5_file_path)

    # Test dataset functionality
    print(f"Dataset size: {len(dataset)}")

    # Fetch a sample
    signal, label = dataset[0]
    print(f"Signal shape: {signal.shape}, Label: {label}")