import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tf_octuplet_loss import OctupletLoss as TFOctupletLoss
from pt_octuplet_loss import OctupletLoss as PTOctupletLoss
import pickle
import torch


with open("/mnt/ssd2/test_embs.pkl", "rb") as f:
    data, labels = pickle.load(f)

tf_loss_fn = TFOctupletLoss(margin=500, metric="euclidean_squared", configuration=[True, True, True, True])
tf_output = tf_loss_fn(data, labels)
print(2660.77685546875 == float(tf_output.numpy()))
#print(1.0572563409805298 == float(tf_output.numpy()))

pt_loss_fn = PTOctupletLoss(margin=500, metric="euclidean_squared", configuration=[True, True, True, True])
pt_output = pt_loss_fn(torch.tensor(data.numpy()[:8]), torch.tensor(labels.numpy()[:4]))
print(float(pt_output.numpy()))



