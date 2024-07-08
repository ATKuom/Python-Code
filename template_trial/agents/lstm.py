import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
import torch.optim as optim
from agents.base import BaseAgent
from models.lstm import LSTM

from ZW_dataset import LstmDataLoader

# import your classes here

# from tensorboardX import SummaryWriter
# from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class LSTM(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = LSTM()

        # define data_loader
        self.data_loader = LstmDataLoader(config=config)

        # define loss
        self.loss = nn.CrossEntropyLoss()

        # define optimizers for both generator and discriminator
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info(
                "WARNING: You have a CUDA device, so you should probably enable CUDA"
            )

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = None

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        pass

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(1, self.config.max_epoch + 1):
            self.train_one_epoch()
            self.validate()

            self.current_epoch += 1

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            # batch_input = padded_train_input[n : n + batch_size]
            # batch_output = train_output[n : n + batch_size]
            # batch_lengths = sequence_lengths[n : n + batch_size]
            # packed_batch_input = nn.utils.rnn.pack_padded_sequence(
            #     batch_input, batch_lengths, batch_first=True, enforce_sorted=False
            # )
            # output = self.model(packed_batch_input.float())
            # loss = self.loss(output, batch_output.argmax(axis=1))
            # train_total += batch_output.size(0)
            # train_correct += (
            #     (output.argmax(axis=1) == batch_output.argmax(axis=1)).sum().item()
            # )
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        self.current_epoch,
                        batch_idx * len(data),
                        len(self.data_loader.train_loader.dataset),
                        100.0 * batch_idx / len(self.data_loader.train_loader),
                        loss.item(),
                    )
                )
            self.current_iteration += 1

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        # batch_input = padded_validation_input[n : n + batch_size]
        # batch_output = validation_output[n : n + batch_size]
        # batch_lengths = validation_lengths[n : n + batch_size]

        # packed_batch_input = nn.utils.rnn.pack_padded_sequence(
        #     batch_input,
        #     batch_lengths,
        #     batch_first=True,
        #     enforce_sorted=False,
        # )
        # # Forward pass
        # output = model(packed_batch_input.float())

        # # Calculate the loss
        # loss += criterion(output, batch_output.argmax(axis=1))
        # total += batch_output.size(0)
        # correct += (
        #     (output.argmax(axis=1) == batch_output.argmax(axis=1)).sum().item()
        # )
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss(
                    output, target, size_average=False
                ).item()  # sum up batch loss
                pred = output.max(1, keepdim=True)[
                    1
                ]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.data_loader.test_loader.dataset)
        self.logger.info(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(self.data_loader.test_loader.dataset),
                100.0 * correct / len(self.data_loader.test_loader.dataset),
            )
        )

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
