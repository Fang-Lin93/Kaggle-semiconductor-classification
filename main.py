import torch
import sys
import pandas as pd
import torch
from dataset import SemiCondData, DataLoader, TestData
from models import MlpMixer, ModelTrainer
from loguru import logger

logger.remove()
logger.add(sys.stderr, level='DEBUG')
logger.add('logs/train.log', level='DEBUG')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_config = {
    'patch_size': -1,
    'channel_dim': 2,
    'num_blocks': 1,
    'fig_size': (267, 275)
}

trainer_config = {
    'tag': 'SemiCond',
    'N_epochs': 100,
    'lr': 0.01,
    'lr_decay': 0.5,
    'device': device,
    'lr_sch_per': 10,
    'l2_regular': 5e-4,
}

"""
Dataset
"""

train_loader = DataLoader(SemiCondData(frac=0.8, train=True, boostrap=True), batch_size=256, shuffle=True)
validate_loader = DataLoader(SemiCondData(frac=0.8, train=False, boostrap=False), batch_size=256, shuffle=True)
test_loader = DataLoader(TestData(), batch_size=256, shuffle=True)

"""
train model
"""
MixMLPModel = MlpMixer(**model_config)
model_trainer = ModelTrainer(MixMLPModel, **trainer_config)
model_trainer.train(validate_loader, validate_loader)
MixMLPModel.save_model()

"""
Prediction
"""
prediction = pd.read_csv('data/submission_sample.csv')
MixMLPModel.load_model()
MixMLPModel.eval()

b_size = test_loader.batch_size
with torch.no_grad():
    for i, x in enumerate(test_loader):
        x = x.to(device)
        score = MixMLPModel(x).softmax(dim=1)
        prediction.loc[i * b_size:(i + 1) * b_size - 1, 'defect_score'] = score[:, 1].numpy()

prediction.to_csv('prediction.csv', index=False)





