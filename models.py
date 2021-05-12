import torch
import time
from torch import nn
from abc import ABC
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

logger.remove()
logger.add('logs/train.log', level='DEBUG')


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


class MlpMixer(nn.Module):
    def __init__(self, patch_size: int, channel_dim: int, num_blocks: int, width: int, height: int):
        super().__init__()
        self.patch_size = patch_size
        self.token_dim = (width // patch_size) * (height // patch_size)
        self.channel_dim = channel_dim
        self.patch_proj = nn.Conv2d(1, channel_dim, kernel_size=(patch_size, patch_size),
                                    stride=(patch_size, patch_size))

        layers = [MixerBlock(self.token_dim, self.channel_dim) for _ in range(num_blocks)]
        self.mixer_mlp_blocks = nn.Sequential(*layers)
        self.out_LayerNorm = nn.LayerNorm([self.token_dim, self.channel_dim])
        self.out_fc = nn.Linear(self.channel_dim, 2)

    def forward(self, x):
        x = self.patch_proj(x).view(-1, self.channel_dim, self.token_dim).transpose(1, 2)
        x = self.mixer_mlp_blocks(x)
        x = self.out_LayerNorm(x)
        x = self.out_fc(x.mean(axis=1))  # add global avg. pooling
        return x

    def save_model(self, file='v0'):
        torch.save(self.state_dict(), f'models_dict/{file}')

    def load_model(self, file='v0'):
        self.load_state_dict(torch.load(f'models_dict/{file}', map_location=torch.device('cpu')))

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
        self.max_acc = -1
        self.min_loss = float('inf')
        self.non_improve = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_regular)

    def loss_function(self, inputs, targets):

        inputs, targets = inputs.to(self.device), targets.to(self.device)
        logits = self.model(inputs)
        return F.cross_entropy(logits, targets), logits.argmax(dim=1)

    def train(self, train_loader, test_loader=None):

        num_batches = len(train_loader)
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
            rec_num = self.train_loop(train_loader, test_loader, epoch, rec_num, num_batches)

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
            loss, pred = self.loss_function(x, y)
            acc = pred.eq(y).sum().true_divide(len(y))
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar(f'Train Loss {self.tag}', loss, rec_num)
            self.writer.add_scalar(f'Train Acc. {self.tag}', acc, rec_num)
            rec_num += 1
            if batch_id % 10 == 0:
                logger.info(f'Train ({self.tag}) [{batch_id}/{num_batches}]'
                            f'({100. * batch_id / num_batches:.0f}%)]'
                            f'\tLoss: {loss.detach():.6f}\tAcc: {100 * acc.detach():.2f}%')
        if val_loader:
            self.validate(val_loader, epoch)
            self.model.train()
        return rec_num

    def validate(self, val_loader, epoch):
        correct = 0
        loss = 0.
        total = 0.
        self.model.eval()
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(val_loader):
                x, y = x.to(self.device), y.to(self.device)
                b_loss, pred = self.loss_function(x, y)
                loss += b_loss
                correct += pred.eq(y).sum()
                total += y.size(0)

            avg_loss, acc = loss / len(val_loader), correct / total

            if avg_loss < self.min_loss:
                self.min_loss = avg_loss
                self.non_improve = 0
            else:
                self.non_improve += 1

            logger.info(f'Validate Loss: {avg_loss:.2f} '
                        f'Validate Acc.: {100 * acc:.2f}%')
            self.writer.add_scalar(f'Validate Loss {self.tag}', avg_loss, epoch)
            self.writer.add_scalar(f'Validate Acc. {self.tag}', acc, epoch)


MixMLPModel = MlpMixer(patch_size=30, channel_dim=20, num_blocks=5, width=267, height=275)


if __name__ == '__main__':
    from dataset import train_loader, validate_loader

    model = MlpMixer(patch_size=30, channel_dim=20, num_blocks=5, width=267, height=275)
    x = torch.rand(2, 1, 267, 275)

    model_trainer = ModelTrainer(model)

    model_trainer.train(train_loader, validate_loader)
