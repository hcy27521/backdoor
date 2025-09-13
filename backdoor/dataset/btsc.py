import requests
from tqdm import tqdm
import os
import subprocess
import functools
from skimage import io, transform
from imageio import imread
import numpy as np
import pandas as pd

from . import dataset, gtsb

CACHE_LOC = dataset.CACHE_LOC

from backdoor.image_utils import ScikitImageArray
from typing import Callable, List, Dict, Tuple

class BTSC(dataset.Dataset):
    """
    Loads the Belgian Traffic Sign Dataset. 
    We return 10 out of 61 classes, chosen for similarity to the GTSC dataset.

    Data Description: https://people.ee.ethz.ch/~timofter/traffic_signs/
    Data Location is also the above page.

    The loaded images are scaled to the desired resolution (default 32x32) before returning.

    Loads in scikit format.
    """
    base_path = os.path.join(CACHE_LOC, "btsc")

    n_classes = 10
    n_channels = 3
    image_shape = (32, 32)

    included_class_ids = [32, 44, 21, 22, 13, 35, 53, 38, 19, 61]
    class_names = ["70km/h", "Oncoming priority", "STOP", "No Entry", "Danger", "Must Turn", "Do not turn", "Bike lane", "Yield", "Do not yield"]

    license = "Data released into the public domain (CC0)"

    # Cherry-pick inheritance from GTSC
    _load_ppm_folder = gtsb.GTSB._load_ppm_folder

    def _download_cache_data(self):
        print('检查本地文件...')
        
        os.makedirs(self.base_path, exist_ok=True)
        
        # 检查文件是否已存在
        training_zip = os.path.join(self.base_path, "BelgiumTSC_Training.zip")
        testing_zip = os.path.join(self.base_path, "BelgiumTSC_Testing.zip")
        
        if not os.path.exists(training_zip):
            print('下载训练集...')
            self._download(self.base_path, "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip")
        else:
            print('训练集 ZIP 文件已存在，跳过下载')
            
        if not os.path.exists(testing_zip):
            print('下载测试集...')
            self._download(self.base_path, "https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip")
        else:
            print('测试集 ZIP 文件已存在，跳过下载')

        # 解压文件
        print('解压文件...')
        if not os.path.exists(f'{self.base_path}/Training/'):
            assert not subprocess.call(['unzip', '-o', training_zip, '-d', self.base_path])
        if not os.path.exists(f'{self.base_path}/Testing/'):
            assert not subprocess.call(['unzip', '-o', testing_zip, '-d', self.base_path])

        # 加载并保存为 npz
        x_train, y_train = self._load_ppm_folder(f'{self.base_path}/Training/')
        x_test, y_test = self._load_ppm_folder(f'{self.base_path}/Testing/')

        np.savez(
            f'{self.base_path}/data.npz',
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test
        )


    def _load_data(self) -> Dict[str, Tuple[ScikitImageArray, np.ndarray]]:
        data = np.load(f'{self.base_path}/data.npz')
        return {'train': dataset.DataTuple((data['x_train'], data['y_train'])), 
                'test': dataset.DataTuple((data['x_test'], data['y_test']))}
        
    # We want the wrapper function to have the right type hint
    get_data: Callable[['BTSC'], Dict[str, Tuple[ScikitImageArray, np.ndarray]]]
    #functools.update_wrapper(super().get_data, _load_data)