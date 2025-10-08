LIBERO_90 dataset tfrecord version.
from /home/v-rusyang/shared_data/dataset/modified_libero_rlds/libero_90_no_noops/*.hdf5, preprocess by openvla data preprocess.

这里要通过某种方法对每一个数据加一个skill/option 标签。

install:

conda env create -f environment_ubuntu.yml
conda activate rlds_env

cd libero_90_w_skill # for example
tfds build -h # for help

``` 
python libero_90_w_skill/add_unique_task_id.py

tfds build --overwrite --data_dir=/mnt/shared_data/dataset/modified_libero_rlds/libero_90_with_skill_id
```