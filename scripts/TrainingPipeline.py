from scripts.CustomDataSet import CustomDataSet
from scripts.CNNTrainer import CNNTrainer
from collections import Counter
from scripts.LabelSplitter import LabelSplitter

from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import os
from types import SimpleNamespace
import math
import logging
from sklearn.model_selection import GroupKFold
import csv
from datetime import datetime
from scripts.StratifiedGroupKFold import StratifiedGroupKFold


class TrainingPipeline:
    def __init__(self, targets):
        self.models_dir = './models'
        self.labels_dir = './labels'
        self.runs_dir = './runs'
        self.targets = targets
        self.cnnTrainer = CNNTrainer()

    def Run(self, params):
        # Read in dataframe
        data_df = pd.read_csv(os.path.join(self.labels_dir, 'data.csv'))

        data_df = data_df.sample(frac=1)

        # Remove non-target columns
        # non_target_cols = list((Counter(['-1', '0', '1', '2', '3']) - Counter(target_cols)).elements())
        # data_df = data_df[data_df.columns.difference(non_target_cols)]

        # Filter (include) observations containing targets
        target_cols = self.targets
        data_df = data_df.loc[(data_df[target_cols] != 0).any(axis=1)]
        data_df = data_df.reset_index(drop=True)

        # Set weights for loss function for imbalanced data
        pos_weights = self.get_class_weights(data_df, target_cols)
        pos_weights = torch.FloatTensor(pos_weights)
        logging.debug("Loss Function Weights: " + ' '.join(map(str, pos_weights)))

        # Set loss criterion (cross entropy for classification where classes > 2)
        criterion = nn.BCELoss(weight=pos_weights)

        # 5-fold cross-validation split by group and stratified by class imbalance
        stratified_group_kfold = StratifiedGroupKFold(n_splits=5)
        # k_fold = GroupKFold(n_splits = 5)

        features = data_df.loc[:, 'id'].to_numpy()

        targets = data_df.loc[:, target_cols].to_numpy()
        targets = np.argmax(targets, axis=1)

        groups = data_df.loc[:, 'subject'].to_numpy()

        results_df = pd.DataFrame(columns=['fold', 'phase', 'epoch', 'loss', 'accuracy'])
        reports_df = pd.DataFrame(columns=['fold', 'metric', 'precision', 'recall', 'f1-score', 'support'])
        conf_matrices_df = pd.DataFrame(columns=['fold', 'act_grade'] + target_cols)
        sum_conf_matrix = np.zeros((params.num_classes, params.num_classes))

        for fold, (train_indices, test_indices) in enumerate(stratified_group_kfold.split(features, targets, groups)):
            logging.debug("")
            logging.debug("Fold # " + str(fold))
            logging.debug("")

            # Define the model for training
            model = torchvision.models.resnet152(pretrained=False)

            # Modify ResNet model to accept multiple channels instead of the default of 3 (RGB)
            model.conv1 = torch.nn.Conv2d(params.in_channel,
                                          64,
                                          kernel_size=(7, 7),
                                          stride=(2, 2),
                                          padding=(3, 3),
                                          bias=False)

            # Modify ResNet model to include regularization:
            model.fc = nn.Sequential(
                nn.Dropout(params.dropout),  # Add dropout to avoid overfitting during training
                nn.Linear(model.fc.in_features, params.num_classes)  # Set # of classes (default - 1000)
            )

            # Define optimizer - optimization algorithm
            optimizer = getattr(optim, params.optimizer)(model.parameters(),
                                                         lr=params.learning_rate,
                                                         weight_decay=params.l2_reg)

            # check to ensure same subject doesn't appear in training and test dataset
            train_group = data_df.iloc[train_indices, :]
            test_group = data_df.iloc[test_indices, :]

            train_group = train_group.loc[:, 'subject'].to_numpy()
            test_group = test_group.loc[:, 'subject'].to_numpy()

            crossover = np.intersect1d(train_group, test_group)

            if crossover.size == 0:
                logging.debug('No Train/Test Subject Leakage Found')
            else:
                logging.debug('Train/Test Subject Leakage: ')
                logging.debug(np.array2string(crossover))

            logging.debug("")

            # check class distribution for training fold
            labels = data_df.iloc[train_indices, :]
            labels = labels.loc[:, target_cols].to_numpy()
            labels = np.argmax(labels, axis=1)

            classes, counts = np.unique(labels, return_counts=True)

            logging.debug('Train')
            for index, _class in enumerate(classes):
                logging.debug('Class: ' + str(target_cols[index]) + ', Count: ' + str(counts[index]))

            logging.debug("")

            # check class distribution for test fold
            labels = data_df.iloc[test_indices, :]
            labels = labels.loc[:, target_cols].to_numpy()
            labels = np.argmax(labels, axis=1)

            classes, counts = np.unique(labels, return_counts=True)

            logging.debug('Test')
            for index, _class in enumerate(classes):
                logging.debug('Class: ' + str(target_cols[index]) + ', Count: ' + str(counts[index]))

            logging.debug("")

            train_dataset = CustomDataSet(data_df.iloc[train_indices, :], target_cols=target_cols,
                                          sequences=params.sequences, augmentation=True)

            train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)

            val_dataset = CustomDataSet(data_df.iloc[test_indices, :], sequences=params.sequences,
                                        target_cols=target_cols, augmentation=False)

            val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True)

            result, report, conf_matrix = self.cnnTrainer.train_network(train_loader=train_loader,
                                                                        val_loader=val_loader,
                                                                        params=params,
                                                                        model=model,
                                                                        criterion=criterion,
                                                                        optimizer=optimizer, labels=target_cols)

            sum_conf_matrix = sum_conf_matrix + conf_matrix

            result_df = pd.DataFrame(result, columns=['phase', 'epoch', 'loss', 'accuracy'])
            result_df['fold'] = str(fold)
            results_df = pd.concat([results_df, result_df]).reset_index(drop=True)

            logging.debug(result_df.to_string())
            logging.debug("")

            report_df = pd.DataFrame.from_dict(report).transpose()
            report_df['metric'] = report_df.index
            cols = list(report_df.columns)
            cols = [cols[-1]] + cols[:-1]
            report_df = report_df[cols]
            report_df = report_df.reset_index(drop=True)
            report_df['fold'] = str(fold)
            reports_df = pd.concat([reports_df, report_df]).reset_index(drop=True)

            logging.debug(report_df.to_string())
            logging.debug("")

            conf_matrix_df = pd.DataFrame(conf_matrix, columns=target_cols)
            conf_matrix_df['act_grade'] = target_cols
            cols = list(conf_matrix_df.columns)
            cols = [cols[-1]] + cols[:-1]
            conf_matrix_df = conf_matrix_df[cols]
            conf_matrix_df['fold'] = str(fold)

            conf_matrices_df = pd.concat([conf_matrices_df, conf_matrix_df]).reset_index(drop=True)

            logging.debug(conf_matrix_df.to_string())
            logging.debug("")

        conf_matrix_df = pd.DataFrame(sum_conf_matrix, columns=target_cols)
        conf_matrix_df['act_grade'] = conf_matrix_df.index
        cols = list(conf_matrix_df.columns)
        cols = [cols[-1]] + cols[:-1]
        conf_matrix_df = conf_matrix_df[cols]
        conf_matrix_df['fold'] = 'sum_folds'

        conf_matrices_df = pd.concat([conf_matrices_df, conf_matrix_df]).reset_index(drop=True)

        self.save_results(params, results_df, reports_df, conf_matrices_df)

    """
    Save the model
    ================== ===========================================================================
    **Arguments:**
    model              the model
    optimizer          the optimizer
    hyperparameters    the hyperparameters
    ================== ===========================================================================
    """
    def save_model(self, model, optimizer, params):
        logging.debug("saving model")
        state_dict = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

        file_path = os.path.join(self.models_dir,
                                'resnet152_'
                                + f'_bs_{params.batch_size}_'
                                + f'lr_{params.learning_rate}'
                                + '.pth.tar')

        torch.save(state_dict, file_path)

    """
    Calculates class weights for an imbalanced dataset. A higher weight means that class has
    a smaller sample size relative to the other classes. This means we will weigh it heavier
    during training.
    ================== ===========================================================================
    **Arguments:**
    labels_df          labels dataframe
    ================== ===========================================================================
    """
    def get_class_weights(self, data_df, target_cols):
        total = data_df.shape[0]
        pos_counts = data_df.loc[:, target_cols].sum().to_list()
        neg_counts = [total - pos_count for pos_count in pos_counts]
        pos_weights = np.ones_like(pos_counts)

        for index, (pos_count, neg_count) in enumerate(zip(pos_counts, neg_counts)):
            pos_weights[index] = math.ceil(neg_count / pos_count)

        return pos_weights

    def objective(self, trial):
        # Set image sequences to train on
        sequences = ['t1.nii.gz', 't1c.nii.gz', 'flair.nii.gz', 't2.nii.gz', 'adc.nii.gz']

        # Set # of image channels (equals number of sequences)
        in_channel = len(sequences)

        # Set # of classes (target tumor grades)
        num_classes = len(self.targets)
        num_epochs = 15

        params = {'sequences': sequences,
                  'in_channel': in_channel,
                  'num_classes': num_classes,
                  'num_epochs': num_epochs,
                  'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                  'learning_rate': trial.suggest_loguniform("learning_rate", 1e-6, 1e-1),
                  'dropout': trial.suggest_uniform("dropout", 0.1, 0.7),
                  'batch_size': trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
                  'l2_reg': trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)}

        params = SimpleNamespace(**params)

        # params.learning_rate = 9.54864588381312e-06
        # params.dropout = 0.176301655747873
        # params.batch_size = 16
        # params.optimizer = "RMSprop"
        # params.l2_reg = 0.000101016739945895

        logging.debug("")
        logging.debug("Hyperparameters: ")
        logging.debug("Sequences: " + str(params.sequences))
        logging.debug("Input Channels: " + str(params.in_channel))
        logging.debug("# Classes: " + str(params.num_classes))
        logging.debug("# Epochs: " + str(params.num_epochs))
        logging.debug("Optimizer: " + str(params.optimizer))
        logging.debug("Learning Rate: " + str(params.learning_rate))
        logging.debug("Dropout: " + str(params.dropout))
        logging.debug("Batch Size: " + str(params.batch_size))
        logging.debug("L2 Regularization: " + str(params.l2_reg))
        logging.debug("")

        self.Run(params)

    """
    Save the model output and results (epoch, accuracy, loss) for each epoch iteration
    ================== ===========================================================================
    **Arguments:**
    hyperparameters    the hyperparameters
    results            list of list of results for each epoch iteration
    ================== ===========================================================================
    """
    def save_results(self, params, results_df, reports_df, conf_matrices_df):
        timestamp = str(datetime.now())

        params = params.__dict__

        with open(self.runs_dir + '/' + timestamp + '_params.csv', 'w') as file:
            writer = csv.DictWriter(file, params.keys())
            writer.writeheader()
            writer.writerow(params)

        results_df.to_csv(os.path.join(self.runs_dir, timestamp + '_results.csv'))
        reports_df.to_csv(os.path.join(self.runs_dir, timestamp + '_reports.csv'))
        conf_matrices_df.to_csv(os.path.join(self.runs_dir, timestamp + '_conf_matrices.csv'))
