#查看标注
import pickle
import numpy as np
f = open('/home/ps/huichenchen/mmdetection3d/data/semantickitti/semantickitti_infos_train.pkl','rb')
data = pickle.load(f)
for line in data:
    print(line)

