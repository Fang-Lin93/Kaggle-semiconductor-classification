
import pandas as pd
import torch
from dataset import test_loader
from models import MixMLPModel
prediction = pd.read_csv('data/submission_sample.csv')
MixMLPModel.load_model()
MixMLPModel.eval()

b_size = test_loader.batch_size
with torch.no_grad():
    for i, x in enumerate(test_loader):
        score = Model(x)
        prediction.loc[i*b_size:(i+1)*b_size-1, 'defect_score'] = score[:, 1].numpy()

prediction.to_csv('prediction.csv', index=False)




