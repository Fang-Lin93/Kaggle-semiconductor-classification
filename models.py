import torch
import time
import os
import torch.nn.functional as F
import pandas as pd
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from dataset import DataLoader, TestData
from loguru import logger
from sklearn import metrics


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        """
        (B, S, C) or (B, C, S)
        """
        return x.transpose(1, 2)


class MlpBlock(nn.Module):
    """
    linear works on the last dim
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.mlp(x)


class MixerBlock(nn.Module):
    def __init__(self, token_dim, channel_dim):
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm([token_dim, channel_dim]),
            Transpose(),
            MlpBlock(token_dim),
            Transpose(),
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm([token_dim, channel_dim]),
            MlpBlock(channel_dim),
        )

    def forward(self, x):
        x = self.token_mixing(x) + x
        return x + self.channel_mixing(x)


class AdjustCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(logits, targets):
        prob = logits.softmax(dim=1)
        entropy = (- prob*prob.log()).sum(dim=1).detach()
        cross_entropy = -prob[torch.arange(prob.size(0)), targets].log()
        return (cross_entropy*entropy).mean()


class StratifiedPatch(nn.Module):
    def __init__(self, channel_dim, fig_size):
        super().__init__()
        self.channel_dim = channel_dim
        self.width, self.height = fig_size
        self.p_32 = nn.Conv2d(1, channel_dim, kernel_size=(32, 32), stride=(32, 32))
        self.p_vertical = nn.Conv2d(1, channel_dim, kernel_size=(16, 128), stride=(16, 128))
        self.p_horizontal = nn.Conv2d(1, channel_dim, kernel_size=(128, 16), stride=(128, 16))

    def forward(self, x):
        # x1 = self.p_16(x).view(x.size(0), self.channel_dim, -1)
        x1 = self.p_32(x).view(x.size(0), self.channel_dim, -1)
        # x3 = self.p_64(x).view(x.size(0), self.channel_dim, -1)
        x2 = self.p_vertical(x).view(x.size(0), self.channel_dim, -1)
        x3 = self.p_horizontal(x).view(x.size(0), self.channel_dim, -1)
        return torch.cat([x1, x2, x3], dim=2)

    def __get_token_dim(self):
        t_dim = sum([(self.width // p_s) * (self.height // p_s) for p_s in [32]])
        t_dim += sum([(self.width // p_x) * (self.height // p_y) for (p_x, p_y) in [(16, 128), (128, 16)]])
        return t_dim

    token_dim = property(__get_token_dim)


class MlpMixer(nn.Module):
    def __init__(self, patch_size: int, channel_dim: int, num_blocks: int, fig_size):
        super().__init__()
        self.patch_size = patch_size
        # self.token_dim = sum([(width // p_s) * (height // p_s) for p_s in [16, 32, 64]])
        self.channel_dim = channel_dim
        self.num_blocks = num_blocks
        if self.patch_size < 0:
            self.patch_proj = StratifiedPatch(self.channel_dim, fig_size)
            self.token_dim = self.patch_proj.token_dim
        else:
            self.patch_proj = nn.Conv2d(1, channel_dim, kernel_size=(patch_size, patch_size),
                                        stride=(patch_size, patch_size))
            width, height = fig_size
            self.token_dim = (width // patch_size) * (height // patch_size)

        layers = [MixerBlock(self.token_dim, self.channel_dim) for _ in range(num_blocks)]
        self.mixer_mlp_blocks = nn.Sequential(*layers)
        self.out_LayerNorm = nn.LayerNorm([self.token_dim, self.channel_dim])
        self.out_fc = nn.Linear(self.channel_dim, 2)

    def feature_vec(self, x):
        with torch.no_grad():
            x = self.patch_proj(x).view(x.size(0), self.channel_dim, -1).transpose(1, 2)
            x = self.mixer_mlp_blocks(x)
            x = self.out_LayerNorm(x)
            return x.mean(axis=1)

    def empirical_label(self, x):
        similarity = nn.CosineSimilarity()
        check_table = torch.zeros(x.size(0), 2)
        with torch.no_grad():
            ftr = self.feature_vec(x)
            check_table[:, 0] = similarity(ftr, self.moving_feature_avg_good.expand(x.size(0), self.channel_dim))
            check_table[:, 1] = similarity(ftr, self.moving_feature_avg_defect.expand(x.size(0), self.channel_dim))
        return check_table.argmax(dim=1)

    def forward(self, x):
        x = self.patch_proj(x).view(x.size(0), self.channel_dim, -1).transpose(1, 2)
        x = self.mixer_mlp_blocks(x)
        x = self.out_LayerNorm(x)
        x = self.out_fc(x.mean(axis=1))  # global avg. pooling

        return x

    def save_model(self, file=''):
        if not os.path.exists('models_dict'):
            os.mkdir('models_dict')
        torch.save(self.state_dict(),
                   f'models_dict/{file}_p{self.patch_size}_c{self.channel_dim}_n{self.num_blocks}.mdl')

    def load_model(self, file=''):
        self.load_state_dict(torch.load(
            f'models_dict/{file}_p{self.patch_size}_c{self.channel_dim}_n{self.num_blocks}.mdl',
            map_location=torch.device('cpu')))

    def __get_num_model_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    num_model_parameters = property(__get_num_model_parameters)


class ModelTrainer:
    def __init__(self, model, **kwargs):

        self.writer = SummaryWriter()
        self.device = kwargs.get('device', torch.device('cpu'))
        self.model = model.to(self.device)
        self.N_epochs = kwargs.get('N_epochs', 10)
        self.lr = kwargs.get('lr', 1e-3)
        self.lr_decay = kwargs.get('lr_decay', 0.5)
        self.lr_sch_per = kwargs.get('lr_sch_per', 10)
        self.tag = kwargs.get('tag', '')
        self.l2_regular = kwargs.get('l2_regular', 1e-3)
        self.tolerance = kwargs.get('tolerance', 5)
        self.early_stop = kwargs.get('early_stop', True)
        self.pred_every = kwargs.get('pred_every', 5)
        self.relabel_start_at = kwargs.get('relabel_start_at', float('inf'))
        self.max_acc = -1
        self.min_loss = float('inf')
        self.non_improve = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_regular)

    def loss_function(self, inputs, targets):

        logits = self.model(inputs)

        # return F.cross_entropy(logits, targets), logits.softmax(dim=1)[:, 1]
        loss = AdjustCrossEntropy()
        return loss(logits, targets), logits.softmax(dim=1)[:, 1]
        # return F.binary_cross_entropy(logits, targets.unsqueeze(1).float()), logits
        # F_loss = FocalLoss(num_classes=2)
        # return F_loss(logits, targets), logits.softmax(dim=1)[:, 1]

    def train(self, tr_loader, val_loader=None):

        num_batches = len(tr_loader)
        lr = self.lr
        self.model.train()
        self.max_acc = -1
        self.min_loss = float('inf')
        self.non_improve = 0

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.l2_regular)
        rec_num = 1
        start_time = time.time()
        for epoch in range(self.N_epochs):
            if epoch % self.lr_sch_per == 0:
                lr *= self.lr_decay
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            rec_num = self.train_loop(tr_loader, val_loader, epoch, rec_num, num_batches)

            logger.info(f' Epoch {epoch}: {(time.time() - start_time):.3f}s, '
                        f'Remaining: {(time.time() - start_time) / (epoch + 1) * (self.N_epochs - (epoch + 1)):.3f}s')

            # Check early stopping!
            if self.early_stop and self.non_improve >= self.tolerance:
                logger.info("Early Stopping!")
                break
        self.model.eval()

    def train_loop(self, tr_loader, val_loader, epoch, rec_num, num_batches):
        logger.info(f'Epoch {epoch} =======================')
        for batch_id, (x, y) in enumerate(tr_loader):
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            relabel = False
            if self.relabel_start_at <= epoch:
                relabel = True
            loss, pred_prob = self.loss_function(x, y)

            pred_class = (pred_prob > 0.5).float().view(-1)
            acc = pred_class.eq(y).sum().true_divide(len(y))
            auc = self.auc(pred_prob.view(-1).detach().cpu().numpy(), y.cpu())

            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar(f'Train Loss {self.tag}', loss, rec_num)
            self.writer.add_scalar(f'Train Acc. {self.tag}', acc, rec_num)
            self.writer.add_scalar(f'Train AUC. {self.tag}', auc, rec_num)
            rec_num += 1
            if batch_id % 10 == 0:
                logger.info(f'Train ({self.tag}) [{batch_id}/{num_batches}]'
                            f'({100. * batch_id / num_batches:.0f}%)]'
                            f'\tLoss: {loss.detach():.6f}'
                            f'\tAcc: {100 * acc.detach():.2f}%'
                            f'\tAUC: {auc:.6f}')
        if val_loader:
            self.validate(val_loader, epoch)
            self.model.train()

        if epoch > 0 and epoch % self.pred_every == 0:
            self.prediction(epoch)
        return rec_num

    def validate(self, val_loader, epoch):
        correct = 0
        loss = 0.
        total = 0.
        auc = 0.
        self.model.eval()
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(val_loader):
                x, y = x.to(self.device), y.to(self.device)
                b_loss, pred_prob = self.loss_function(x, y)

                pred_class = (pred_prob > 0.5).float().view(-1)
                auc += self.auc(pred_prob.view(-1).cpu().numpy(), y.cpu())

                loss += b_loss
                correct += pred_class.eq(y).sum()
                total += y.size(0)

            avg_loss, acc = loss / len(val_loader), correct / total
            avg_auc = auc / len(val_loader)

            # if avg_loss < self.min_loss:
            #     self.min_loss = avg_loss
            #     self.non_improve = 0
            # else:
            #     self.non_improve += 1
            if avg_auc > self.max_acc:
                self.max_acc = avg_auc
                self.non_improve = 0
            else:
                self.non_improve += 1

            logger.info(f'Validate Loss: {avg_loss:.2f} '
                        f'Validate Acc.: {100 * acc:.2f}% '
                        f'Validate AUC: {avg_auc:.6f}')
            self.writer.add_scalar(f'Validate Loss {self.tag}', avg_loss, epoch)
            self.writer.add_scalar(f'Validate Acc. {self.tag}', acc, epoch)
            self.writer.add_scalar(f'Train AUC. {self.tag}', avg_auc, epoch)

    @staticmethod
    def auc(pred_y, y):
        """
        AUC
        """
        fpr, tpr, threshold = metrics.roc_curve(y, pred_y)
        return metrics.auc(fpr, tpr)

    def prediction(self, epoch):
        """
        Predicted csv
        """
        prediction = pd.read_csv('data/submission_sample.csv')
        self.model.eval()
        test_loader = DataLoader(TestData(), batch_size=512, shuffle=False)
        b_size = test_loader.batch_size
        with torch.no_grad():
            for i, x in enumerate(test_loader):
                x = x.to(self.device)
                score = self.model(x).softmax(dim=1)
                prediction.loc[i * b_size:(i + 1) * b_size - 1, 'defect_score'] = score[:, 1].cpu().numpy()
        if not os.path.exists('results'):
            os.mkdir('results')
        prediction.to_csv(f'results/pred_{epoch}.csv', index=False)


if __name__ == '__main__':

    model_ = MlpMixer(patch_size=30, channel_dim=20, num_blocks=5, fig_size=(267, 275))
    x_ = torch.rand(2, 1, 267, 275)
