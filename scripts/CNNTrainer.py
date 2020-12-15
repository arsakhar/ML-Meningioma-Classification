from torch.utils.data import Dataset
import os
import torch
import pandas as pd
import torch.nn as nn
import logging
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import csv
import numpy as np


class CNNTrainer(Dataset):
    def __init__(self):
        self.models_dir = './models'
        self.runs_dir = './runs'
        self.cuda_device = torch.device("cuda:0")

    def train_network(self, train_loader, val_loader, params, model, criterion, optimizer, labels):
        if torch.cuda.device_count() > 1:
            logging.debug("Using Parallel CUDA GPUs")
            logging.debug("")

            model = nn.DataParallel(model, device_ids=[0,6,7])

        model.to(self.cuda_device)
        criterion.to(self.cuda_device)

        data_loaders = {'train': train_loader, 'val': val_loader}

        results = []
        for epoch in range(params.num_epochs):
            logging.debug(f"Epoch {epoch} / {params.num_epochs}")
            logging.debug("--------------")

            # we alternate between training and testing
            for phase in ['train', 'val']:
                losses = []

                if phase == 'train':
                    model.train()

                else:
                    model.eval()

                for batch_idx, (data, targets) in enumerate(data_loaders[phase]):
                    # get data to cuda if possible
                    data = data.to(device=self.cuda_device, dtype=torch.float)

                    # get labels to cuda if possible
                    targets = targets.to(device=self.cuda_device)

                    # forward pass
                    # probs represents the probabilities for each class for the batch of images
                    probs = model(data)

                    # compute loss function
                    loss = criterion(probs, targets)

                    # store loss for each batch in list
                    losses.append(loss.item())

                    # zero the parameter (weight) gradients
                    optimizer.zero_grad()

                    if phase == 'train':
                        # backward pass
                        loss.backward()

                        # optimizer step
                        optimizer.step()

                # calculate average loss for phase
                mean_loss = sum(losses) / len(losses)
                logging.debug(f"{phase} Loss: {mean_loss}")

                # calculate accuracy for phase
                accuracy = self.check_accuracy(data_loaders[phase], model)
                logging.debug(f"{phase} Accuracy: {accuracy}")

                results.append([phase, epoch, mean_loss, accuracy])
                del probs, loss

            logging.debug("--------------")
            logging.debug("")

        report = self.get_classification_report(data_loaders['val'], model, labels)
        conf_matrix = self.get_confusion_matrix(data_loaders['val'], model, labels)

        return results, report, conf_matrix

    def get_confusion_matrix(self, loader, model, labels):
        model.eval()

        targets = []
        preds = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.cuda_device, dtype=torch.float)
                y = y.to(device=self.cuda_device)

                # get probabilities that a label is associated with image for all images in batch
                probs = model(x)

                # take the label with the highest probability as the prediction
                _, pred = probs.max(1)

                targets = targets + y.tolist()
                preds = preds + pred.tolist()

        conf_matrix = confusion_matrix(y_true=targets, y_pred=preds, labels=None)

        return conf_matrix

    def get_classification_report(self, loader, model, labels):
        model.eval()

        targets = []
        preds = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.cuda_device, dtype=torch.float)
                y = y.to(device=self.cuda_device)

                # get probabilities that a label is associated with image for all images in batch
                probs = model(x)

                # take the label with the highest probability as the prediction
                _, pred = probs.max(1)

                targets = targets + y.tolist()
                preds = preds + pred.tolist()

        report = classification_report(y_true=targets, y_pred=preds,
                                       target_names=labels,
                                       output_dict=True)

        return report

    """
    Check the accuracy of the model at the end of an epoch
    ================== ===========================================================================
    **Arguments:**
    loader             data loader - contains image batch and associated labels
    model              the trained model
    
    **Returns:**
    accuracy           accuracy of the trained model
    ================== ===========================================================================
    """
    def check_accuracy(self, loader, model):
        num_correct_predictions = 0
        num_predictions = 0
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.cuda_device, dtype=torch.float)
                y = y.to(device=self.cuda_device)

                # get probabilities that a label is associated with image for all images in batch
                probs = model(x)

                # take the label with the highest probability as the prediction
                _, pred = probs.max(1)

                # compare the predictions with the actual labels for images in batch
                num_correct_predictions += (pred == y).sum()

                # keep running tally of number of images
                num_predictions += pred.size(0)

            # calculate accuracy for train/val phase
            accuracy = round(float(num_correct_predictions) / float(num_predictions) * 100, 2)

            return accuracy