import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import math
#from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot
import logging

from keras.datasets import mnist
from keras.datasets import cifar10

from pathlib import Path
import requests
import pickle
import gzip

import torch
import math
import torch.nn.functional as F
from torch import linalg as LA
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import random
import math

from datetime import datetime
import yaml
import os
import argparse

import multiprocessing as mp
from keras.datasets import cifar10
from operator import itemgetter


def worker(c, num_of_epochs):
    logging.debug("Launching training for client: {}".format(c.id))
    c.call_training(num_of_epochs)


class Net2nn(nn.Module):
    def __init__(self, input_size, n_classes):
        super(Net2nn, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def clone(self):
        model_clone = Net2nn(self.input_size, self.n_classes)
        model_clone.fc1.weight.data = self.fc1.weight.data.clone()
        model_clone.fc2.weight.data = self.fc2.weight.data.clone()
        model_clone.fc3.weight.data = self.fc3.weight.data.clone()
        model_clone.fc1.bias.data = self.fc1.bias.data.clone()
        model_clone.fc2.bias.data = self.fc2.bias.data.clone()
        model_clone.fc3.bias.data = self.fc3.bias.data.clone()
        return model_clone


class CNN(nn.Module):
    def __init__(self, rgb_channels, n_classes, dataset):
        super(CNN, self).__init__()

        self.rgb_channels = rgb_channels
        self.n_classes = n_classes
        self.dataset = dataset

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=rgb_channels, out_channels=16,
                              kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32,
                              kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1
        if self.dataset == 'mnist':
            self.fc1 = nn.Linear(32 * 5 * 5, self.n_classes)
        elif self.dataset == 'cifar':
            self.fc1 = nn.Linear(32 * 6 * 6, self.n_classes)

    def forward(self, x, to_print=False):
        # Set 1
        if to_print:
            print('INPUT', x.shape)
        out = self.cnn1(x)
        if to_print:
            print('CNN1', out.shape)
        out = self.relu1(out)
        out = self.maxpool1(out)
        if to_print:
            print('MAXPOOL1', out.shape)

        # Set 2
        out = self.cnn2(out)
        if to_print:
            print('CNN2', out.shape)

        out = self.relu2(out)

        out = self.maxpool2(out)
        if to_print:
            print("after the 2nd maxpool:{} ".format(out.shape))
        # Flatten
        out = out.view(out.size(0), -1)
        if to_print:
            print("after the flatten:{} ".format(out.shape))
        out = self.fc1(out)
        if to_print:
            print('FINAL', out.shape)
        return out

    def clone(self):
        model_clone = CNN(self.rgb_channels, self.n_classes, self.dataset)
        model_clone.cnn1.weight.data = self.cnn1.weight.data.clone()
        model_clone.cnn2.weight.data = self.cnn2.weight.data.clone()
        model_clone.fc1.weight.data = self.fc1.weight.data.clone()
        model_clone.cnn1.bias.data = self.cnn1.bias.data.clone()
        model_clone.cnn2.bias.data = self.cnn2.bias.data.clone()
        model_clone.fc1.bias.data = self.fc1.bias.data.clone()
        return model_clone


class CNN_2(nn.Module):   
    def __init__(self, rgb_channels, n_classes, dataset):
        
        super(CNN_2, self).__init__()
        self.input_fc_layer = 0
        self.rgb_channels = rgb_channels
        self.n_classes = n_classes
        self.dataset = dataset
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=rgb_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

        )
        if self.dataset == 'mnist':
            self.input_fc_layer = 6272
        elif self.dataset == 'cifar':
            self.input_fc_layer = 8192

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.input_fc_layer, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, self.n_classes)
        )
    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # fc layer
        x = self.fc_layer(x)

        return x

    def clone(self):
        model_clone = CNN_2(self.rgb_channels, self.n_classes, self.dataset)
        model_clone.load_state_dict(self.state_dict())
        return model_clone


class Setup:
    '''Read the dataset, instantiate the clients and the server
       It receive in input the path to the data, and the number of clients
    '''

    def __init__(self, conf_file):
        self.conf_file = conf_file

        self.settings = self.load(self.conf_file)

        self.data_path = self.settings['setup']['data_path']
        self.n_clients = self.settings['setup']['n_clients']
        self.learning_rate = self.settings['setup']['learning_rate']
        self.num_of_epochs = self.settings['setup']['num_of_epochs']
        self.batch_size = self.settings['setup']['batch_size']
        self.momentum = self.settings['setup']['momentum']
        self.random_clients = self.settings['setup']['random_clients']
        self.federated_runs = self.settings['setup']['federated_runs']
        self.saving_dir = self.settings['setup']['save_dir']
        self.multiprocessing = self.settings['setup']['multiprocessing']
        self.to_save = self.settings['setup']['to_save']
        self.network_type = self.settings['setup']['network_type']
        self.dataset = self.settings['setup']['dataset']
        self.aggregate_function = 'average'
        self.virtual_clients = 0

        if 'virtual_clients' in self.settings['setup'].keys():
            self.virtual_clients = self.settings['setup']['virtual_clients']

        if 'aggregation' in self.settings['setup'].keys():
            self.aggregate_function = self.settings['setup']['aggregation']

        self.saved = False

        if "saved" not in self.settings.keys():
            self.start_time = datetime.now()
        else:
            self.saved = True
            self.start_time = datetime.strptime(
                self.settings['saved']['timestamp'], '%Y%m%d%H%M')

        timestamp = self.start_time.strftime("%Y%m%d%H%M")
        self.path = os.path.join(self.saving_dir, timestamp)

        logging.debug("Setup: creating client with path %s (%s)",
                      self.path, self.saved)

        self.list_of_clients = []
        self.X_train, self.y_train, self.X_test, self.y_test, self.rgb_channels, self.height, self.width, self.n_classes = self.__load_dataset()

        self.create_clients()

        self.server = Server(self.list_of_clients, self.random_clients, self.virtual_clients,
                             self.learning_rate, self.num_of_epochs,
                             self.batch_size, self.momentum,
                             self.saved, self.path, self.multiprocessing, self.network_type, self.dataset,
                             self.rgb_channels, self.height, self.width, self.n_classes)

    def load(self, conf_file):
        with open(conf_file) as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
            return settings

    def run(self, federated_runs=None):
        if federated_runs != None:
            self.federated_runs = federated_runs
        for i in range(self.federated_runs):
            logging.info(
                "Setup: starting run of the federated learning number %s", (i + 1))
            self.server.training_clients()
            if self.aggregate_function == 'average':
                self.server.update_averaged_weights()
            elif self.aggregate_function == 'krum':
                self.server.update_krum_weights()
            self.server.send_weights()
            for idx_client,c in enumerate(self.list_of_clients):
                logging.info('Current {} accuracy: {}'.format(c.id,self.server.get_accuracy(idx_client)))

    def __load_dataset(self):
        X_train, y_train, X_test, y_test = None, None, None, None
        if self.dataset == 'cifar':
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        elif self.dataset == 'mnist':
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
        else:
            logging.error('no dataset info loaded')
        # flat the images as 1d feature vector.
        # mnist 28x28x1 - cifar10 32x32x3
        rgb_channels = 1
        if len(X_train.shape) == 4:
            rgb_channels = X_train.shape[-1]

        height = X_train.shape[1]
        width = X_train.shape[2]

        classes = len(np.unique(y_train))

#         X_train = X_train.reshape(X_train.shape[0], reduce((lambda x, y: x * y), list(X_train.shape)[1:]))
#         X_test = X_test.reshape(X_test.shape[0], reduce((lambda x, y: x * y), list(X_train.shape)[1:]))
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        # Original data is uint8 (0-255). Scale it to range [0,1].
        X_train /= 255
        X_test /= 255

        if y_test.shape != (len(y_test),):
            y_test = y_test.reshape(len(y_test))
        if y_train.shape != (len(y_train),):
            y_train = y_train.reshape(len(y_train))

        return X_train, y_train, X_test, y_test, rgb_channels, height, width, classes

    def __create_iid_datasets(self):
        # 1. randomly shuffle both train and test sets
        X_train, y_train = shuffle(self.X_train, self.y_train, random_state=42)
        X_test, y_test = shuffle(self.X_test, self.y_test, random_state=42)
        # 2. split evenly both train and test sets by the number of clients
        X_trains = np.array_split(X_train, self.n_clients)
        y_trains = np.array_split(y_train, self.n_clients)
        X_tests = np.array_split(X_test, self.n_clients)
        y_tests = np.array_split(y_test, self.n_clients)

        return X_trains, y_trains, X_tests, y_tests

    def create_clients(self, iid=True):

        if iid:
            X_trains, y_trains, X_tests, y_tests = self.__create_iid_datasets()
        else:
            X_trains, y_trains, X_tests, y_tests = self.__create_non_iid_datasets()

        for i in range(self.n_clients):
            c = Client(str(i), X_trains[i], y_trains[i], X_tests[i], y_tests[i],
                       self.learning_rate, self.num_of_epochs, self.batch_size,
                       self.momentum, self.saved, self.path, self.network_type, self.dataset,
                       self.rgb_channels, self.height, self.width, self.n_classes
                       )
            self.list_of_clients.append(c)

    def add_clients(self, client):
        self.list_of_clients.append(client)
        client.update_model_weights(self.server.main_model)

    def save_models(self):
        self.server.save_model(self.path)
        for c in self.list_of_clients:
            c.save_model(self.path)

    def save(self):
        timestamp = self.start_time.strftime("%Y%m%d%H%M")
        self.path = os.path.join(self.saving_dir, timestamp)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.save_models()
        self.settings['saved'] = {"timestamp": timestamp}
        with open(os.path.join(self.path, 'setup.yaml'), 'w') as fout:
            yaml.dump(self.settings, fout)


class Server:
    '''
    The Server class owns the central model.
    - It initializes the main model and it updates the weights to the clients
    - It handles the training among the clients
    - It receives the weights from clients and it averages them for its main
      model updating it
    '''

    def __init__(self, list_of_clients, random_clients, virtual_clients=0,
                 learning_rate=0.01, num_of_epochs=10,
                 batch_size=32, momentum=0.9,
                 saved=False, path=None, multiprocessing=0,
                 network_type='NN', dataset='mnist', rgb_channels=1,
                 height=28, width=28, n_classes=10):

        self.list_of_clients = list_of_clients
        self.random_clients = random_clients
        self.virtual_clients = virtual_clients
        self.learning_rate = learning_rate
        self.num_of_epochs = num_of_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.multiprocessing = multiprocessing
        self.network_type = network_type
        self.dataset = dataset

        self.rgb_channels = rgb_channels
        self.width = width
        self.height = height
        self.n_classes = n_classes

        self.current_anomalous_client = None

        self.selected_clients = []
        if self.network_type == 'NN':
            self.main_model = Net2nn(self.rgb_channels*self.width*self.height, self.n_classes)
        elif self.network_type == 'CNN':
            # logging.info('network_type: {}'.format(self.network_type))
            self.main_model = CNN(self.rgb_channels, self.n_classes, self.dataset)
        elif self.network_type == 'CNN2':
            # logging.info('network_type: {}'.format(self.network_type))
            self.main_model = CNN_2(self.rgb_channels, self.n_classes, self.dataset)


        if saved:
            self.main_model.load_state_dict(
                torch.load(os.path.join(path, "main_model")))

        self.main_optimizer = torch.optim.SGD(
            self.main_model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.main_criterion = nn.CrossEntropyLoss()
        self.send_weights()

    def hstack_w_b(self, a, b):
        return np.hstack((a, b.reshape(-1, 1)))

    def get_all_weights(self):
        # for each client tuples of weights for each layer (C:(X1,X2,X3)) where X1 = Weights + bias
        weights = {}
        for c in self.selected_clients:
            w_1 = c.model.fc1.weight.data.clone().numpy()
            b_1 = c.model.fc1.bias.data.clone().numpy()
            w_2 = c.model.fc2.weight.data.clone().numpy()
            b_2 = c.model.fc2.bias.data.clone().numpy()
            w_3 = c.model.fc3.weight.data.clone().numpy()
            b_3 = c.model.fc3.bias.data.clone().numpy()
            x_1 = self.hstack_w_b(w_1, b_1)
            x_2 = self.hstack_w_b(w_2, b_2)
            x_3 = self.hstack_w_b(w_3, b_3)

            weights[c.id] = (x_1, x_2, x_3)
        return weights

    def compute_distance(self, l_i, l_j):

        return np.linalg.norm(l_i-l_j)

    def get_layer_distances(self, w_c_i, w_c_j):
        distances = []
        for l in range(len(w_c_i)):
            #             print(w_c_i[l],w_c_j[l])
            #             print('Distance:',np.linalg.norm(w_c_i[l]-w_c_j[l]))
            distances.append(np.linalg.norm(w_c_i[l]-w_c_j[l]))
#             distances.append(self.compute_distance(w_c_i[l],w_c_j[l]))

        return distances

    def get_closests(self, weights, thr):
        closests = {}
        for c_i in weights.keys():
            pair_distances = {}
            for c_j in weights.keys():
                if c_i != c_j:
                    distances = self.get_layer_distances(weights[c_i], weights[c_j])
#                     print(c_i,c_j,distances)
                    for l, _ in enumerate(distances):
                        if l not in pair_distances.keys():
                            pair_distances[l] = {}
                        pair_distances[l][c_j] = distances[l]
            c_closest = []
            for l in pair_distances.keys():
                top_k = sorted(pair_distances[l].items(), key=lambda x: x[1], reverse=False)[:thr]
                c_closest.append(
                    sorted(pair_distances[l].items(), key=lambda x: x[1], reverse=False)[:thr])
            closests[c_i] = c_closest
        return closests

    def get_krum_scores(self, closests):
        scores = {}
        for c in closests.keys():
            for l, _ in enumerate(closests[c]):
                if l not in scores.keys():
                    scores[l] = {}
                scores[l][c] = np.sum([c_j[1]*c_j[1] for c_j in closests[c][l]])
        return scores

    def get_weights_biases_krum(self, scores, weights):
        matrices = []
        kr_clients = []
        for l in scores.keys():
            c = sorted(scores[l].items(), key=itemgetter(1), reverse=False)[0][0]
            matrix = weights[c][l]
            weight = matrix[:, :-1]
            bias = matrix[:, -1]
            matrices.append([weight, bias])
            kr_clients.append(c)
        return matrices, kr_clients

    def hstack_w_b(self, a, b):
        return np.hstack((a, b.reshape(-1, 1)))

    def update_krum_weights(self, f=1):
        # get all the weights and biases from each client c_i=(w1,w2,w3)
        weights = self.get_all_weights()
#         print("Client 0 ",weights['client_0'])
#         print("Client 1 ",weights['client_1'])
        thr = max(2, len(self.selected_clients) - f - 2)
#         print('Threshold {}'.format(thr))
        # for each client the n-f-2 closest ids of other clients
        closests = self.get_closests(weights, thr)
#         print("Client 0 ",closests['client_0'])
#         print("Client 0 ",closests['client_1'])
        scores = self.get_krum_scores(closests)
        logging.debug('krum scores: {}'.format(scores))
        matrices, kr_clients = self.get_weights_biases_krum(scores, weights)
        logging.info('server: krum selected client for updates: {}'.format(kr_clients))
        with torch.no_grad():
            self.main_model.fc1.weight.data = torch.from_numpy(matrices[0][0])
            self.main_model.fc2.weight.data = torch.from_numpy(matrices[1][0])
            self.main_model.fc3.weight.data = torch.from_numpy(matrices[2][0])

            self.main_model.fc1.bias.data = torch.from_numpy(matrices[0][1])
            self.main_model.fc2.bias.data = torch.from_numpy(matrices[1][1])
            self.main_model.fc3.bias.data = torch.from_numpy(matrices[2][1])

    def get_current_anomalous_client(self):
        return self.current_anomalous_client

    def send_weights(self):
        for c in self.list_of_clients:
            c.update_model_weights(self.main_model)
            # c.get_distance()
            # logging.info('Distance after send_weights c: {} distance: {:.3f}'.format(c.id, c.distance))

    def select_clients(self):
        # print('Clients: {}'.format(self.list_of_clients))

        self.attackers = [c for c in self.list_of_clients if (
            'Receiver' in c.id) or ('Sender' in c.id)]
        self.clients = [c for c in self.list_of_clients if ('Observer' not in c.id) and (
            'Receiver' not in c.id) and ('Sender' not in c.id)]
        logging.debug('Server: N. attackers {}'.format(len(self.attackers)))
        logging.debug('Server: N. clients {}'.format(len(self.clients)))
        n_clients = len(self.clients)
        types_of_clients = ['a', 'c']
        others_prob = 1 - (self.random_clients * 2)
        sel_first = random.choices(types_of_clients, weights=[
                                   2 * self.random_clients, others_prob], k=n_clients)
        n_others = max(sel_first.count('c'), 10-len(self.attackers))
        n_attackers = min(sel_first.count('a'), len(self.attackers))
        logging.info('N_selected others {} and attackers {}'.format(n_others, n_attackers))
        # better to keep here to check everytime the list of clients
        return random.sample(self.attackers, n_attackers) + random.sample(self.clients, n_others)

    def training_clients(self):
        logging.debug("Server: training_clients()")
        if self.virtual_clients:
            self.selected_clients = self.select_clients()
        else:
            list_of_eligibles = [c for c in self.list_of_clients if c.id != "client_Observer"]
            self.selected_clients = random.sample(list_of_eligibles, math.floor(
                len(self.list_of_clients) * self.random_clients))

        logging.info("Server: selected clients {}".format([c.id for c in self.selected_clients]))

        if self.multiprocessing:
            processes = []
            for i, c in enumerate(self.selected_clients):
                p = mp.Process(target=worker, args=(c, self.num_of_epochs))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
        else:
            for c in self.selected_clients:
                c.call_training(self.num_of_epochs)
                c.get_distance()
                # logging.info("Client {}: distance {:.3f}".format(c.id, c.distance) )

    def get_averaged_weights_nn(self):

        fc1_mean_weight = torch.zeros(
            size=self.list_of_clients[0].model.fc1.weight.shape)
        fc1_mean_bias = torch.zeros(
            size=self.list_of_clients[0].model.fc1.bias.shape)

        fc2_mean_weight = torch.zeros(
            size=self.list_of_clients[0].model.fc2.weight.shape)
        fc2_mean_bias = torch.zeros(
            size=self.list_of_clients[0].model.fc2.bias.shape)

        fc3_mean_weight = torch.zeros(
            size=self.list_of_clients[0].model.fc3.weight.shape)
        fc3_mean_bias = torch.zeros(
            size=self.list_of_clients[0].model.fc3.bias.shape)

        with torch.no_grad():
            for c in self.selected_clients:
                logging.debug("Server: getting weights for %s", c.id)
                fc1_mean_weight += c.model.fc1.weight.data.clone()
                fc1_mean_bias += c.model.fc1.bias.data.clone()

                fc2_mean_weight += c.model.fc2.weight.data.clone()
                fc2_mean_bias += c.model.fc2.bias.data.clone()

                fc3_mean_weight += c.model.fc3.weight.data.clone()
                fc3_mean_bias += c.model.fc3.bias.data.clone()

            fc1_mean_weight = fc1_mean_weight / len(self.selected_clients)
            fc1_mean_bias = fc1_mean_bias / len(self.selected_clients)

            fc2_mean_weight = fc2_mean_weight / len(self.selected_clients)
            fc2_mean_bias = fc2_mean_bias / len(self.selected_clients)

            fc3_mean_weight = fc3_mean_weight / len(self.selected_clients)
            fc3_mean_bias = fc3_mean_bias / len(self.selected_clients)

        return fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias

    def get_averaged_weights_cnn(self):

        cnn1_mean_weight = torch.zeros(
            size=self.list_of_clients[0].model.cnn1.weight.shape)
        cnn1_mean_bias = torch.zeros(
            size=self.list_of_clients[0].model.cnn1.bias.shape)

        cnn2_mean_weight = torch.zeros(
            size=self.list_of_clients[0].model.cnn2.weight.shape)
        cnn2_mean_bias = torch.zeros(
            size=self.list_of_clients[0].model.cnn2.bias.shape)

        fc1_mean_weight = torch.zeros(
            size=self.list_of_clients[0].model.fc1.weight.shape)
        fc1_mean_bias = torch.zeros(
            size=self.list_of_clients[0].model.fc1.bias.shape)

        with torch.no_grad():
            for c in self.selected_clients:
                logging.debug("Server: getting weights for %s", c.id)
                cnn1_mean_weight += c.model.cnn1.weight.data.clone()
                cnn1_mean_bias += c.model.cnn1.bias.data.clone()

                cnn2_mean_weight += c.model.cnn2.weight.data.clone()
                cnn2_mean_bias += c.model.cnn2.bias.data.clone()

                fc1_mean_weight += c.model.fc1.weight.data.clone()
                fc1_mean_bias += c.model.fc1.bias.data.clone()

            cnn1_mean_weight = cnn1_mean_weight / len(self.selected_clients)
            cnn1_mean_bias = cnn1_mean_bias / len(self.selected_clients)

            cnn2_mean_weight = cnn2_mean_weight / len(self.selected_clients)
            cnn2_mean_bias = cnn2_mean_bias / len(self.selected_clients)

            fc1_mean_weight = fc1_mean_weight / len(self.selected_clients)
            fc1_mean_bias = fc1_mean_bias / len(self.selected_clients)

        return cnn1_mean_weight, cnn1_mean_bias, cnn2_mean_weight, cnn2_mean_bias, fc1_mean_weight, fc1_mean_bias

    def get_averaged_weights_cnn_2(self):
        status_dict = {}
        with torch.no_grad():
            for name in self.main_model.state_dict().keys():
                data = torch.zeros(
                    size = self.main_model.state_dict()[name].shape)
                for c in self.selected_clients:
                    m = c.model 
                    data += m.state_dict()[name].data.clone()
                data = data / len(self.selected_clients)
                status_dict[name] = data
        return status_dict

    def get_anomalous_client(self, norm="fro"):
        """
            - `norm` refers to the type of the norm to be calculated between the two weight matrices:
              by default this is "fro", which stands for Frobenius norm (a.k.a. L2-norm)
              Other possible values are those indicated by the numpy API's: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
              e.g., 1, np.inf, "nuc", etc.
        """

        # 1. get current global model parameters
        with torch.no_grad():
            # current_server_fc1 = self.main_model.fc1.weight.data
            # current_server_fc2 = self.main_model.fc2.weight.data
            # current_server_fc3 = self.main_model.fc3.weight.data

            max_distance = 0
            c_avg_distance = 0

            # 2. check the client with the largest average distance w.r.t. the current global model

            for c in self.selected_clients:
                if c.distance > max_distance:
                    self.current_anomalous_client = c.id
                    max_distance = c.distance

            # for c in self.selected_clients:
            #     c_avg_distance += np.linalg.norm(current_server_fc1 -
            #                                      c.model.fc1.weight.data, ord=norm)
            #     c_avg_distance += np.linalg.norm(current_server_fc2 -
            #                                      c.model.fc2.weight.data, ord=norm)
            #     c_avg_distance += np.linalg.norm(current_server_fc3 -
            #                                      c.model.fc3.weight.data, ord=norm)

            #     c_avg_distance = c_avg_distance / 3

            #     if c_avg_distance > max_distance:
            #         self.current_anomalous_client = c.id
            #         max_distance = c_avg_distance

            logging.info("Server: current anomalous client is {:s} [{:s} distance = {:.3f}]".format(
                self.current_anomalous_client, norm, max_distance))

    # def get_anomalous_client_cnn(self, norm="fro"):
    #     """
    #         - `norm` refers to the type of the norm to be calculated between the two weight matrices:
    #           by default this is "fro", which stands for Frobenius norm (a.k.a. L2-norm)
    #           Other possible values are those indicated by the numpy API's: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    #           e.g., 1, np.inf, "nuc", etc.
    #     """

    #     # 1. get current global model parameters
    #     with torch.no_grad():
    #         current_server_cnn1 = self.main_model.cnn1.weight.data
    #         current_server_cnn2 = self.main_model.cnn2.weight.data
    #         current_server_fc1 = self.main_model.fc1.weight.data

    #         max_distance = 0
    #         c_avg_distance = 0

    #         # 2. check the client with the largest average distance w.r.t. the current global model
    #         for c in self.selected_clients:
    #             c_avg_distance += np.linalg.norm(current_server_cnn1 -
    #                                              c.model.cnn1.weight.data, ord=norm)
    #             c_avg_distance += np.linalg.norm(current_server_cnn2 -
    #                                              c.model.cnn2.weight.data, ord=norm)
    #             c_avg_distance += np.linalg.norm(current_server_fc1 -
    #                                              c.model.fc1.weight.data, ord=norm)

    #             c_avg_distance = c_avg_distance / 3

    #             if c_avg_distance > max_distance:
    #                 self.current_anomalous_client = c.id
    #                 max_distance = c_avg_distance

    #         logging.info("Server: current anomalous is {:s} [{:s} distance = {:.3f}]".format(
    #             self.current_anomalous_client, norm, max_distance))

    def update_averaged_weights(self):
        self.get_anomalous_client()
        if self.network_type == 'NN':
            self.update_averaged_weights_nn()
        elif self.network_type == 'CNN':
            self.update_averaged_weights_cnn()
        elif self.network_type == 'CNN2':
            self.main_model = self.update_averaged_weights_cnn_2()
        else:
            logging.error('NETWORK TYPE NOT SUPPORTED')

    def update_averaged_weights_nn(self):
        fc1_mean_weight, fc1_mean_bias, fc2_mean_weight, fc2_mean_bias, fc3_mean_weight, fc3_mean_bias = self.get_averaged_weights_nn()
        with torch.no_grad():
            self.main_model.fc1.weight.data = fc1_mean_weight.data.clone()
            self.main_model.fc2.weight.data = fc2_mean_weight.data.clone()
            self.main_model.fc3.weight.data = fc3_mean_weight.data.clone()

            self.main_model.fc1.bias.data = fc1_mean_bias.data.clone()
            self.main_model.fc2.bias.data = fc2_mean_bias.data.clone()
            self.main_model.fc3.bias.data = fc3_mean_bias.data.clone()

    def update_averaged_weights_cnn(self):
        cnn1_mean_weight, cnn1_mean_bias, cnn2_mean_weight, cnn2_mean_bias, fc1_mean_weight, fc1_mean_bias = self.get_averaged_weights_cnn()
        with torch.no_grad():
            self.main_model.cnn1.weight.data = cnn1_mean_weight.data.clone()
            self.main_model.cnn2.weight.data = cnn2_mean_weight.data.clone()
            self.main_model.fc1.weight.data = fc1_mean_weight.data.clone()

            self.main_model.cnn1.bias.data = cnn1_mean_bias.data.clone()
            self.main_model.cnn2.bias.data = cnn2_mean_bias.data.clone()
            self.main_model.fc1.bias.data = fc1_mean_bias.data.clone()

    def update_averaged_weights_cnn_2(self):
        status_dict = self.get_averaged_weights_cnn_2()
        with torch.no_grad():
            self.main_model.load_state_dict(status_dict)
        return self.main_model

    def predict(self, data):
        # to be fixed how to reshape the values
        self.main_model.eval()
        with torch.no_grad():
            if (self.network_type == 'CNN' or self.network_type == 'CNN2') and self.dataset == 'cifar':
                data = data.permute(0, 3, 1, 2)
            elif (self.network_type == 'CNN' or self.network_type == 'CNN2') and self.dataset == 'mnist':
                data = data.reshape(-1, 1, 28, 28)
            elif self.network_type == 'NN':
                data = data.reshape(-1, self.rgb_channels*self.width*self.height)
            return self.main_model(data)

    def save_model(self, path):
        out_path = os.path.join(path, "main_model")
        torch.save(self.main_model.state_dict(), out_path)

    def get_accuracy(self, idx_client=0):
        if idx_client < len(self.list_of_clients):
            test_ds = TensorDataset(self.list_of_clients[idx_client].x_test, self.list_of_clients[idx_client].y_test)
            test_dl = DataLoader(test_ds, batch_size=self.batch_size)

            return self.list_of_clients[idx_client].validation(test_dl)[1]
            # self.list_of_clients[idx_client].train_accuracy
        else:
            return None


class Client:
    '''A client who has its own dataset to use for training.
       The main methods of the Client class are:
       - Load the data
       - Get the weights from the server
       - Train the model
       - Return the weights to the server
       - Get a sample and perform a prediction with the probabilities for each class
        '''

    def __init__(self, id, x_train, y_train, x_test, y_test, learning_rate=0.01,
                 num_of_epochs=10, batch_size=32, momentum=0.9, saved=False,
                 path=None, network_type='NN', dataset='mnist', rgb_channels=1, height=28, width=28, n_classes=10):

        logging.debug("Client: __init__()")
        self.id = "client_" + id
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.network_type = network_type
        self.dataset = dataset
        self.rgb_channels = rgb_channels
        self.height = height
        self.width = width
        self.n_classes = n_classes
        self.main_model = None
        self.distance = 0
        # training and test can be splitted inside the client class
        # now we are passing them while instantiate the class

        logging.debug('{} {} {} {}'.format(self.rgb_channels,
                                           self.height, self.width, self.n_classes))

        logging.debug("Client: x_train : %s = %s | y_train : %s = %s", type(
            x_train), x_train.shape, type(y_train), y_train.shape)

        x_train, y_train, x_test, y_test = map(
            torch.tensor, (x_train, y_train, x_test, y_test))
        y_train = y_train.type(torch.LongTensor)
        y_test = y_test.type(torch.LongTensor)

        # Adjust the shape of training and test for the proper net.

        x_train, x_test = self.reshape_dataset(x_train, x_test)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model_name = "model" + self.id
        if self.network_type == 'NN':
            self.model = Net2nn(self.rgb_channels*self.width*self.height, self.n_classes)
        elif self.network_type == 'CNN':
            self.model = CNN(self.rgb_channels, self.n_classes, self.dataset)
        elif self.network_type == 'CNN2':
            self.model = CNN_2(self.rgb_channels, self.n_classes, self.dataset)
        else: 
            logging.error('NETWORK TYPE NOT SUPPORTED')
            exit(-1)

        logging.debug("Client: %s | %s | %s", self.id, saved, path)

        if saved:
            id_model = int(id) % 10
            self.model.load_state_dict(torch.load(
                os.path.join(path, "model_client_{}".format(id_model))))

        self.optimizer_name = "optimizer" + str(self.id)
        self.optimizer_info = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        self.criterion_name = "criterion" + str(self.id)
        self.criterion_info = nn.CrossEntropyLoss()
        self.test_accuracy = 0
        self.train_accuracy = 0
        self.train_loss = 1
        self.test_loss = 1

    def reshape_dataset(self, x_train, x_test):
        if (self.network_type == 'CNN' or self.network_type == 'CNN2') and self.dataset == 'cifar':
            # to be adjusted with cifar10 since the dimensions are different
            x_train = x_train.permute(0, 3, 1, 2)
            x_test = x_test.permute(0, 3, 1, 2)
        elif (self.network_type == 'CNN' or self.network_type == 'CNN2') and self.dataset == 'mnist':
            x_train = x_train.reshape(-1, 1, 28, 28)
            x_test = x_test.reshape(-1, 1, 28, 28)
        elif self.network_type == 'NN':
            x_train = x_train.reshape(-1, self.rgb_channels*self.width*self.height)
            x_test = x_test.reshape(-1, self.rgb_channels*self.width*self.height)

        return x_train, x_test

    def update_model_weights(self, main_model):
        self.main_model = main_model
        if self.network_type == 'NN':
            self.update_model_weights_nn(main_model)
        elif self.network_type == 'CNN':
            self.update_model_weights_cnn(main_model)
        elif self.network_type == 'CNN2':
            self.update_model_weights_cnn_2(main_model)
        else: 
            logging.error('NETWORK TYPE NOT SUPPORTED')

    def update_model_weights_nn(self, main_model):
        with torch.no_grad():
            self.model.fc1.weight.data = main_model.fc1.weight.data.clone()
            self.model.fc2.weight.data = main_model.fc2.weight.data.clone()
            self.model.fc3.weight.data = main_model.fc3.weight.data.clone()
            self.model.fc1.bias.data = main_model.fc1.bias.data.clone()
            self.model.fc2.bias.data = main_model.fc2.bias.data.clone()
            self.model.fc3.bias.data = main_model.fc3.bias.data.clone()

    def update_model_weights_cnn(self, main_model):
        with torch.no_grad():
            self.model.cnn1.weight.data = main_model.cnn1.weight.data.clone()
            self.model.cnn2.weight.data = main_model.cnn2.weight.data.clone()
            self.model.fc1.weight.data = main_model.fc1.weight.data.clone()
            self.model.cnn1.bias.data = main_model.cnn1.bias.data.clone()
            self.model.cnn2.bias.data = main_model.cnn2.bias.data.clone()
            self.model.fc1.bias.data = main_model.fc1.bias.data.clone()

    def update_model_weights_cnn_2(self, main_model):
        with torch.no_grad():
            self.model.load_state_dict(main_model.state_dict())

    def call_training(self, n_of_epoch):
        train_ds = TensorDataset(self.x_train, self.y_train)
        train_dl = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True)

        test_ds = TensorDataset(self.x_test, self.y_test)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size * 2)

        for epoch in range(n_of_epoch):

            self.train_loss, self.train_accuracy = self.train(train_dl)
            self.test_loss, self.test_accuracy = self.validation(test_dl)
            logging.debug("Client: {}".format(self.id) + " | epoch: {:3.0f}".format(epoch + 1) +
                          " | train accuracy: {:7.5f}".format(self.train_accuracy) + " | test accuracy: {:7.5f}".format(self.test_accuracy))

    def train(self, train_dl):
        logging.debug("INSIDE THE TRAINING CLIENT {}".format(self.id))
        self.model.train()
        train_loss = 0.0
        correct = 0
        device = 'cpu'
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.model.to(device)
        for data, target in train_dl:
            data = data.to(device)
            target = target.to(device)
            output = self.model(data)
            # logging.debug("N. OF CORRECT INSTANCES: {}".format(correct))
            loss = self.criterion_info(output, target)
            self.optimizer_info.zero_grad()
            loss.backward()
            self.optimizer_info.step()

            train_loss += loss.item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

        return train_loss / len(train_dl), correct / len(train_dl.dataset)

    def validation(self, test_dl):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        device = 'cpu'
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.model.to(device)

        with torch.no_grad():
            for data, target in test_dl:
                data = data.to(device)
                target = target.to(device)

                output = self.model(data)

                test_loss += self.criterion_info(output, target).item()
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)
                                         ).sum().item()

        test_loss /= len(test_dl)
        correct /= len(test_dl.dataset)

        return (test_loss, correct)

    def get_distance_nn(self, norm='fro'):
        with torch.no_grad():
            current_server_fc1 = self.main_model.fc1.weight.data
            current_server_fc2 = self.main_model.fc2.weight.data
            current_server_fc3 = self.main_model.fc3.weight.data

            c_avg_distance = 0

            # 2. check the client with the largest average distance w.r.t. the current global model
            c_avg_distance += np.linalg.norm(current_server_fc1 -
                                             self.model.fc1.weight.data, ord=norm)
            c_avg_distance += np.linalg.norm(current_server_fc2 -
                                             self.model.fc2.weight.data, ord=norm)
            c_avg_distance += np.linalg.norm(current_server_fc3 -
                                             self.model.fc3.weight.data, ord=norm)

            c_avg_distance = c_avg_distance / 3

            self.distance = c_avg_distance
            return self.distance

    def get_distance_cnn(self, norm='fro'):
        with torch.no_grad():
            current_server_cnn1 = self.main_model.cnn1.weight.data
            current_server_cnn2 = self.main_model.cnn2.weight.data
            current_server_fc1 = self.main_model.fc1.weight.data

            c_avg_distance = 0

            # 2. check the client with the largest average distance w.r.t. the current global model
            c_avg_distance += LA.norm(current_server_cnn1 -
                                      self.model.cnn1.weight.data)
            c_avg_distance += LA.norm(current_server_cnn2 -
                                      self.model.cnn2.weight.data)
            c_avg_distance += LA.norm(current_server_fc1 -
                                      self.model.fc1.weight.data)

            c_avg_distance = c_avg_distance / 3

            self.distance = c_avg_distance
            return self.distance
    def get_distance_cnn_2(self, norm='fro'):
        c_avg_distance = 1
        # for name in self.model.state_dict().keys():
        #     c_avg_distance += LA.norm(self.main_model.state_dict()[name].data -self.model.state_dict()[name].data)
        # c_avg_distance = c_avg_distance / len(self.model.state_dict().keys())

        self.distance = c_avg_distance
        return self.distance


    def get_distance(self, norm='fro'):
        if self.network_type == 'NN':
            return self.get_distance_nn(norm)
        elif self.network_type == 'CNN':
            return self.get_distance_cnn(norm)
        elif self.network_type == 'CNN2':
            return self.get_distance_cnn_2(norm)
        else:
            logging.error('NETWORK NOT SUPPORTED')
            exit(-1)



    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            if self.network_type == 'CNN' and self.dataset == 'cifar':
                data = data.permute(0, 3, 1, 2)
            elif self.network_type == 'CNN' and self.dataset == 'mnist':
                data = data.reshape(-1, 1, 28, 28)
            elif self.network_type == 'NN':
                data = data.reshape(-1, self.rgb_channels*self.width*self.height)
            return self.model(data)

    def save_model(self, path):
        out_path = os.path.join(path, "model_{}".format(self.id))
        torch.save(self.model.state_dict(), out_path)


if __name__ == '__main__':
    logging.basicConfig(
        format='[+] %(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("conf_file", type=str)
    args = parser.parse_args()
    logging.debug("Running with configuration file {}".format(args.conf_file))

    conf_file = args.conf_file

    setup = Setup(conf_file)
    setup.run()
    if setup.to_save:
        setup.save()
