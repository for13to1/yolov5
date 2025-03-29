# setup

```shell
conda install pytorch torchvision torchaudio -c pytorch
# 安装 pytorch 时即已安装依赖: numpy pillow pyyaml requests sympy
conda install -c conda-forge ultralytics
# 安装 ultralytics 时即已安装依赖:
# psutil scipy opencv ffmpeg pandas
# seaborn # conda-forge
conda install gitpython
conda install -c conda-forge pycocotools
pip install --no-cache-dir --no-deps thop
# pip install --no-cache-dir --no-deps ultralytics-thop # 会报 warning
```

跑通

```shell
python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images
# 执行过程中会尝试 pip install --no-cache-dir "gitpython>=3.1.30"
# 其中 --no-cache-dir 的意思是 不将下载的文件保存到本地缓存目录中

python val.py --weights yolov5s.pt --data coco.yaml --img 640 --half
# 执行过程中会尝试 pip install --no-cache-dir "pycocotools>=2.0.6"

python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov5s.pt --cache
# 执行过程中会尝试 pip install --no-cache-dir "thop>=0.1.1"
```
