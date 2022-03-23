from federated_learning import Setup, Client
import random
import argparse
import logging
import numpy
import enum
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import signal
import pandas
import sys
import baseliner as bl
import jpg_baseliner as jbl
import random

from chi_squared_test import chi_squared_test
from datetime import datetime
import yaml
import os
import subprocess



# something strange with RNG
random.seed()

MNIST_SEARCH_THREASHOLD = 1 / (28 * 28)
MNIST_SIZE = 60000

CIFAR_SEARCH_THREASHOLD = 1 / (32 * 32)
CIFAR_SIZE = 60000

#NTRAIN = 1  # rounds of training
#NTRANS = 10  # rounds for transmission tests
DELTA = 0.1
ALPHA = 0.5
BATCH_SIZE = 32
NSELECTION = 3
DELTA_PLT_X = 1
DELTA_PLT_Y = 1

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }

SCORE_LOG = 'scoreL.csv'
EVENT_LOG = 'eventL.csv'

score_dict = {
    'X': [],
    'Y': []
}
event_dict = {
    'X': [],
    'E': []
}

save_path = ""

def increase_error_rate(error_rate):
    error_rate += 1
    return error_rate

timer = 0

def log_score(y):
    global timer
    score_dict['X'].append(timer)
    score_dict['Y'].append(y)


def log_event(e):
    global timer
    event_dict['X'].append(timer)
    event_dict['E'].append(e)

def update_plot(y):
    global timer
    hl.set_xdata(numpy.append(hl.get_xdata(), [timer]))
    hl.set_ydata(numpy.append(hl.get_ydata(), [y]))


def add_vline():
    global timer
    plt.axvline(x=timer)

def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def create_sample(image,dataset):
    x_train = numpy.array([image])
    x_train = x_train.astype('float32')
    x_train /= 255
    x_train_torch = torch.from_numpy(x_train[[0]])
    if dataset == 'cifar':
        x_train_torch = x_train_torch.reshape(1,32,32,3).permute(0, 3, 1, 2)
    elif dataset == 'mninst':
        x_train_torch = x_train_torch.reshape(1,28,28,1).flatten()
    # return x_train[[0]]
    return x_train_torch

def create_samples(images, dataset):
    l = []
    for i in range(len(images)):
        l.append(create_sample(images[i],dataset))
    return l


class ReceiverState(enum.Enum):
    Crafting = 1
    Calibrating = 2
    Ready = 3
    Transmitting = 4


class Sender(Client):

    def __init__(self, images, labels, n_channels, frame, pattern, network_type, dataset, rgb_channels, height, width):
        self.bit = [None]*n_channels
        self.n_channels = n_channels
        self.sent = False
        self.frame_count = -1
        self.frame = frame
        self.pattern = pattern
        self.frame_start = [0]*n_channels
        self.labels = labels
        x_train = numpy.array(images)
        x_train = x_train.astype('float32')
        x_train /= 255
        x_train = torch.from_numpy(x_train).reshape(-1,height,width,rgb_channels)
        # logging.info('SHAPE X TRAIN SENDER: {}'.format(x_train.shape))
        # logging.info('TYPE Y TRAIN SENDER: {}'.format(labels[0]))
        # y_train = labels
        y_train = numpy.array(labels)
        y_train = torch.from_numpy(y_train)

        super().__init__("Sender", x_train, y_train, x_train, y_train, network_type=network_type, dataset=dataset, rgb_channels=rgb_channels, height=height, width=width)

    # Covert channel send
    def call_training(self, n_of_epoch):
        logging.debug("Sender: call_training()")
        # super().call_training(n_of_epoch)
        self.send_to_model(n_of_epoch)

    def update_model_weights(self, main_model):
        logging.debug("Sender: update_model_weights()")
        super().update_model_weights(main_model)

        logging.debug("Sender: frame_count = %s", self.frame_count)

        if self.frame_count == 0:
            # x_pred = torch.from_numpy(self.x_train[[0]])
            for c in range(self.n_channels):
                logging.info('SHAPE CRAFT EXAMPLE TO PREDICT ON SENDER: {}'.format(self.x_train[[c]].shape))
                self.frame_start[c] = self.label_predict(self.x_train[[c]])

                if self.pattern is None:
                    self.bit[c] = random.randint(0, 1)
                else:
                    self.bit[c] = int(self.pattern[0])
                    self.pattern =  self.pattern[1:] + self.pattern[0:1]

            logging.info("Sender: frame starts with %s", self.frame_start)
            logging.info("Sender: SENDING %s", self.bit)
            log_event("Sent " + str(self.bit))

        self.frame_count = (self.frame_count + 1) % self.frame

    def label_predict(self, x_pred):
        prediction = self.predict(x_pred)
        logging.debug("Sender: prediction %s", prediction)
        # TODO: must return max element only
        return torch.argmax(prediction)

    # forces biases to transmit one bit through the model
    def send_to_model(self, n_of_epoch):

        for c in range(self.n_channels):
            logging.info('SHAPE CRAFT EXAMPLE TO PREDICT ON SENDER: {}'.format(self.x_train[[c]].shape))

            pred = self.label_predict(self.x_train[[c]])
            logging.info("Sender: start = %s, current %s", self.frame_start[c], pred)
            if self.bit[c] == 1 and self.frame_start[c] == pred:
                # change prediction
                logging.info("Sender: channel %s switching from %s", c, self.frame_start[c])

                if self.frame_start[c] == self.y_train[c][0]:
                    logging.debug("Sender: %s == %s", self.frame_start[c], self.y_train[c][0])
                    y_train_trans = self.labels[c][1]
                else:
                    logging.debug("Sender: %s != %s", self.frame_start[c], self.y_train[c][0])
                    y_train_trans = self.labels[c][0]

                logging.debug("Sender: index %s", y_train_trans)
                logging.debug("Sender: x_train %s", self.x_train[c])
                # bias injection dataset

                x_train_reshaped = self.x_train[c]
                """
                if self.network_type == 'CNN':
                    x_train_reshaped = self.x_train[c].reshape(-1, 1, 28, 28)
                elif self.network_type == 'NN':
                    x_train_reshaped = self.x_train[c].reshape(1,784)
                else:
                    logging.error("Sender: unsupported network type")
                """

                train_ds = TensorDataset(x_train_reshaped, torch.from_numpy(numpy.array([y_train_trans])))
                train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)

                # bias testing dataset
                test_ds = TensorDataset(x_train_reshaped, torch.from_numpy(numpy.array([y_train_trans])))
                test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

                for epoch in range(n_of_epoch):
                    train_loss, train_accuracy = self.train(train_dl)
                    test_loss, test_accuracy = self.validation(test_dl)

            elif self.bit[c] == 0 and not self.frame_start[c] == pred:
                logging.info("Sender: channel %s restoring %s", c, self.frame_start[c])

                if self.frame_start[c] == self.y_train[c][0]:
                    logging.debug("Sender: %s == %s", self.frame_start[c], self.y_train[c][0])
                    y_train_trans = self.labels[c][0]
                else:
                    logging.debug("Sender: %s != %s", self.frame_start[c], self.y_train[c][0])
                    y_train_trans = self.labels[c][1]

                logging.debug("Sender: index %s", y_train_trans)
                logging.debug("Sender: x_train %s", self.x_train[c])
                # bias injection dataset

                x_train_reshaped = self.x_train[c]
                """
                if self.network_type == 'CNN':
                    x_train_reshaped = self.x_train[c].reshape(-1, 1, 28, 28)
                elif self.network_type == 'NN':
                    x_train_reshaped = self.x_train[c].reshape(1,784)
                else:
                    logging.error("Sender: unsupported network type")
                """

                train_ds = TensorDataset(x_train_reshaped, torch.from_numpy(numpy.array([y_train_trans])))
                train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)

                # bias testing dataset
                test_ds = TensorDataset(x_train_reshaped, torch.from_numpy(numpy.array([y_train_trans])))
                test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

                for epoch in range(n_of_epoch):
                    train_loss, train_accuracy = self.train(train_dl)
                    test_loss, test_accuracy = self.validation(test_dl)
            else:
                logging.info("Sender: channel already set")

class Receiver(Client):

    def __init__(self,n_channels,frame_size,network_type, dataset, rgb_channels, height, width):
        self.bit = [None]*n_channels
        self.n_channels = n_channels
        self.images = [None]*n_channels
        self.labels = [None]*n_channels
        self.selection_count = 0
        self.frame = frame_size
        self.frame_count = 0
        self.frame_start = [0]*n_channels
        self.frame_end = [0]*n_channels
        self.state = ReceiverState.Crafting
        x_train = numpy.random.random(size=[1,height,width,rgb_channels])
        y_train = numpy.random.random(size=[1,height,width,rgb_channels])
        x_train = x_train.astype('float32')
        super().__init__("Receiver", x_train, y_train, x_train, y_train,network_type=network_type, dataset=dataset, rgb_channels=rgb_channels, height=height, width=width)

    def call_training(self, n_of_epoch):
        logging.debug("Receiver: call_training()")

        if self.state == ReceiverState.Calibrating:

            self.selection_count += 1
            logging.info("Receiver: selected %s times", self.selection_count)
            if self.selection_count > NSELECTION:
                self.state = ReceiverState.Ready
        else:
            pass

    # Covert channel receive
    def update_model_weights(self, main_model):
        logging.debug("Receiver: update_model_weights()")
        super().update_model_weights(main_model)

        logging.debug("Receiver: frame_count = %s", self.frame_count)

        if self.state == ReceiverState.Crafting:
            self.craft()
        elif self.state == ReceiverState.Calibrating:
            self.calibrate()
        else:  # self.state == ReceiverState.Transmitting:
            self.read_from_model()

    def label_predict(self, x_pred):
        prediction = self.predict(x_pred)
        logging.debug("Receiver: prediction %s", prediction)
        return torch.argmax(prediction)

    def read_from_model(self):

        for c in range(self.n_channels):
            x_pred = create_sample(self.images[c],self.dataset)            
            # x_train = numpy.array([self.images[c]])
            # x_train = x_train.astype('float32')
            # x_train /= 255

            # x_pred = torch.from_numpy(x_train[0])
            pred = self.label_predict(x_pred)

            if self.frame_count == 0:
                self.frame_start[c] = pred
            elif self.frame_count == self.frame - 1:
                self.frame_end[c] = pred

                if self.frame_start[c] == self.frame_end[c]:
                    self.bit[c] = 0
                else:
                    self.bit[c] = 1
            else:
                pass

        if self.frame_count == 0:
            logging.info("Receiver: frame starts with = %s", self.frame_start)
        elif self.frame_count == self.frame - 1:
            logging.info("Receiver: frame ends with = %s", self.frame_end)
            logging.info("Receiver: RECEIVED: %s", self.bit)
            log_event("Received " + str(self.bit))

        self.frame_count = (self.frame_count + 1) % self.frame

    def calibrate(self):
        self.frame += 1

    def dataset_size(self):
        if self.dataset == 'mnist':
            return MNIST_SIZE
        elif self.dataset == 'cifar':
            return CIFAR_SIZE
        else:
            logging.error("Unsupported dataset %s", self.dataset)
            return 0

    def craft(self):

        logging.info("Sender: crafting channel samples")

        c = 0
        #allocated = []

        while c < self.n_channels:

            size = self.dataset_size()
            c += self.search_edge_example(c, size)

        logging.info("Receiver: channels ready")

        if self.frame < 1:
            self.state = ReceiverState.Calibrating
        else:
            self.state = ReceiverState.Ready

    def search_edge_example(self, c, size):

        i = random.randint(0, size-1)
        j = random.randint(0, size-1)

        logging.debug("Receiver: trying to craft from %s %s", i, j)

        if self.dataset == 'mnist':
            image_i = bl.linearize(bl.get_image(i))
            image_j = bl.linearize(bl.get_image(j))

            i_label = self.label_predict(create_sample(image_i, self.dataset))

            imageH = bl.hmix(image_i, image_j, ALPHA)
            H_label = self.label_predict(create_sample(imageH, self.dataset))

            alpha, y0_label, y1_label = self.hsearch(image_i, image_j, i_label, H_label, 0, ALPHA)

            if alpha > 0: # and not y0_label in allocated and not y1_label in allocated:
                logging.info("Receiver: found hmix(%s, %s, %s) = %s | %s", i, j, alpha, y0_label, y1_label)
                self.images[c] = bl.hmix(image_i, image_j, alpha)
                self.labels[c] = [y0_label.cpu(), y1_label.cpu()]
                #allocated.append(y0_label)
                #allocated.append(y1_label)
                return 1
            else:
                logging.debug("Receiver: not found for (%s,%s)", i, j)

            imageV = bl.vmix(image_i, image_j, ALPHA)
            V_label = self.label_predict(create_sample(imageV, self.dataset))

            alpha, y0_label, y1_label = self.vsearch(image_i, image_j, i_label, V_label, 0, ALPHA)

            if alpha > 0: # and not y0_label in allocated and not y1_label in allocated:
                logging.info("Receiver: found vmix(%s, %s, %s) = %s | %s", i, j, alpha, y0_label, y1_label)
                self.images[c] = bl.vmix(image_i, image_j, alpha)
                self.labels[c] = [y0_label.cpu(), y1_label.cpu()]
                #allocated.append(y0_label)
                #allocated.append(y1_label)
                return 1
            else:
                logging.debug("Receiver: not found for (%s,%s)", i, j)
            return 0

        elif self.dataset == 'cifar':
            image_i = jbl.linearize(jbl.get_image(i))
            image_j = jbl.linearize(jbl.get_image(j))

            i_label = self.label_predict(create_sample(image_i, self.dataset))

            imageH = jbl.hmix(image_i, image_j, ALPHA)
            
            H_label = self.label_predict(create_sample(imageH, self.dataset))

            alpha, y0_label, y1_label = self.hsearch(image_i, image_j, i_label, H_label, 0, ALPHA)

            if alpha > 0: # and not y0_label in allocated and not y1_label in allocated:
                logging.info("Receiver: found hmix(%s, %s, %s) = %s | %s", i, j, alpha, y0_label, y1_label)
                self.images[c] = jbl.hmix(image_i, image_j, alpha)
                self.labels[c] = [y0_label.cpu(), y1_label.cpu()]
                #allocated.append(y0_label)
                #allocated.append(y1_label)
                return 1
            else:
                logging.debug("Receiver: not found for (%s,%s)", i, j)

            imageV = jbl.vmix(image_i, image_j, ALPHA)
            V_label = self.label_predict(create_sample(imageV, self.dataset))

            alpha, y0_label, y1_label = self.vsearch(image_i, image_j, i_label, V_label, 0, ALPHA)

            if alpha > 0: # and not y0_label in allocated and not y1_label in allocated:
                logging.info("Receiver: found vmix(%s, %s, %s) = %s | %s", i, j, alpha, y0_label, y1_label)
                self.images[c] = jbl.vmix(image_i, image_j, alpha)
                self.labels[c] = [y0_label.cpu(), y1_label.cpu()]
                #allocated.append(y0_label)
                #allocated.append(y1_label)
                return 1
            else:
                logging.debug("Receiver: not found for (%s,%s)", i, j)
            return 0
        else:
            logging.error("Unknown dataset %s", self.dataset)
        return 0

    def hsearch(self, image_i, image_j, y0_label, y1_label, alpha_min, alpha_max):

        logging.debug("H-searching between %s and %s", y0_label, y1_label)

        if self.dataset == 'mnist':
            if y0_label == y1_label:
                return -1,None,None

            if alpha_max < alpha_min + MNIST_SEARCH_THREASHOLD:
                return alpha_min, y0_label, y1_label

            imageM = bl.hmix(image_i, image_j, (alpha_min + alpha_max) / 2)
            yM_label = self.label_predict(create_sample(imageM, self.dataset))
            if y0_label != yM_label:
                return self.hsearch(image_i, image_j, y0_label, yM_label, alpha_min, (alpha_min + alpha_max) / 2)
            else:
                return self.hsearch(image_i, image_j, yM_label, y1_label, (alpha_min + alpha_max) / 2, alpha_max)
        elif self.dataset == 'cifar':
            if y0_label == y1_label:
                return -1,None,None

            if alpha_max < alpha_min + CIFAR_SEARCH_THREASHOLD:
                return alpha_min, y0_label, y1_label

            imageM = jbl.hmix(image_i, image_j, (alpha_min + alpha_max) / 2)
            yM_label = self.label_predict(create_sample(imageM, self.dataset))
            if y0_label != yM_label:
                return self.hsearch(image_i, image_j, y0_label, yM_label, alpha_min, (alpha_min + alpha_max) / 2)
            else:
                return self.hsearch(image_i, image_j, yM_label, y1_label, (alpha_min + alpha_max) / 2, alpha_max)
        else:
            logging.error("Unknown dataset %s", self.dataset)

    def vsearch(self, image_i, image_j, y0_label, y1_label, alpha_min, alpha_max):

        logging.debug("V-searching between %s and %s", y0_label, y1_label)

        if self.dataset == 'mnist':
            if y0_label == y1_label:
                return -1,None,None

            if alpha_max < alpha_min + MNIST_SEARCH_THREASHOLD:
                return alpha_min, y0_label, y1_label

            imageM = bl.vmix(image_i, image_j, (alpha_min + alpha_max) / 2)
            yM_label = self.label_predict(create_sample(imageM, self.dataset))
            if y0_label != yM_label:
                return self.vsearch(image_i, image_j, y0_label, yM_label, alpha_min, (alpha_min + alpha_max) / 2)
            else:
                return self.vsearch(image_i, image_j, yM_label, y1_label, (alpha_min + alpha_max) / 2, alpha_max)
        elif self.dataset == 'cifar':
            if y0_label == y1_label:
                return -1,None,None

            if alpha_max < alpha_min + CIFAR_SEARCH_THREASHOLD:
                return alpha_min, y0_label, y1_label

            imageM = jbl.vmix(image_i, image_j, (alpha_min + alpha_max) / 2)
            yM_label = self.label_predict(create_sample(imageM, self.dataset))
            if y0_label != yM_label:
                return self.vsearch(image_i, image_j, y0_label, yM_label, alpha_min, (alpha_min + alpha_max) / 2)
            else:
                return self.vsearch(image_i, image_j, yM_label, y1_label, (alpha_min + alpha_max) / 2, alpha_max)
        else:
            logging.error("Unknown dataset %s", self.dataset)

class Observer(Client):

    def __init__(self,network_type, dataset, rgb_channels, height, width):
        self.frame_count = 0
        self.frame = 0
        self.samples = None
        x_train = numpy.random.random(size=[1,height,width,rgb_channels])
        y_train = numpy.random.random(size=[1,height,width,rgb_channels])
        # x_train = numpy.array([])
        # y_train = numpy.array([])
        x_train = x_train.astype('float32')
        super().__init__("Observer", x_train, y_train, x_train, y_train,network_type=network_type,
                        dataset=dataset, rgb_channels=rgb_channels, height=height, width=width
                        )

    # Covert channel send
    def call_training(self, n_of_epoch):
        pass

    def set_frame(self, frame):
        self.frame = frame

    def set_sample(self, s):
        self.samples = s

    def update_model_weights(self, main_model):
        logging.debug("Observer: update_model_weights()")
        super().update_model_weights(main_model)

        if self.samples != None:
            pred = []
            for c in range(len(self.samples)):
                pred.append(self.predict(self.samples[c]))
                # update_plot(torch.argmax(pred))

            logging.debug("Observer: global prediction = %s, frame_count = %s", pred, self.frame_count)
            log_score(pred)

        if self.frame > 0:
            if self.frame_count == 0:
                add_vline()
                log_event('Frame start')
            self.frame_count = (self.frame_count + 1) % self.frame

        global timer
        timer += 1


class Setup_env:
    '''Setup simulation environment from YAML configuration file.
    '''

    def __init__(self, conf_file):
        self.conf_file = conf_file

        self.settings = self.load(self.conf_file)

        self.save_tests = self.settings['setup']['save_tests']
        self.saving_tests_dir = self.settings['setup']['tests_dir']
        self.prob_selection = self.settings['setup']['random_clients']
        self.batch_size = self.settings['setup']['batch_size']
        self.n_bits = self.settings['setup']['n_bits']
        self.n_train_offset = self.settings['setup']['n_train_offset']
        self.n_Rcal = self.settings['setup']['n_Rcal']
        self.network_type = self.settings['setup']['network_type']
        self.docker = True
        self.saved = False

        if "n_channels" in self.settings['setup'].keys():
            self.n_channels = self.settings['setup']['n_channels']
        else:
            self.n_channels = 1

        if "pattern" in self.settings['setup'].keys():
            self.pattern = str(self.settings['setup']['pattern'])
        else:
            self.pattern = None

        if "frame_size" in self.settings['setup'].keys():
            self.frame_size = self.settings['setup']['frame_size']
        else:
            self.frame_size = 0

        if "dataset" in self.settings['setup'].keys():
            self.dataset = self.settings['setup']['dataset']
        else:
            self.dataset = 'mnist'

        if self.dataset == 'mnist':
            self.width = 28
            self.height = 28
            self.rgb_channels = 1
        elif self.dataset == 'cifar':
            self.width = 32
            self.height = 32
            self.rgb_channels = 3
        else: 
            logging.error('Dataset not implemented yet!')

        if "docker" in self.settings['setup'].keys():
            self.docker = self.settings['setup']['docker']

        if "saved" not in self.settings.keys():
            self.start_time = datetime.now()
        else:
            self.saved = True
            self.start_time = datetime.strptime(
                self.settings['saved']['timestamp'], '%Y%m%d%H%M%S')

        timestamp = self.start_time.strftime("%Y%m%d%H%M%S")
        self.path = os.path.join(self.saving_tests_dir, timestamp)

    def load(self, conf_file):
        with open(conf_file) as f:
            settings = yaml.safe_load(f)
            return settings

    def save(self):
        id_folder = None
        if self.docker:
            id_folder = subprocess.check_output('cat /proc/self/cgroup | grep "docker" | sed  s/\\\\//\\\\n/g | tail -1', shell=True).decode("utf-8").rstrip()
        else:
            id_folder = str(os.getpid())
        timestamp = self.start_time.strftime("%Y%m%d%H%M%S")
        self.path = os.path.join(self.saving_tests_dir, id_folder)
        global save_path
        save_path = self.path
        logging.info("save path %s", save_path)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.settings['saved'] = {"timestamp": timestamp}
        self.settings['saved'] = {"id container": id_folder}
        with open(os.path.join(self.path, 'setup_tests.yaml'), 'w') as fout:
            yaml.dump(self.settings, fout)

    def id_tests(self):
        timestamp = self.start_time.strftime("%Y%m%d%H%M%S")
        id_tests = "Score-attack_" + "p_" + str(self.prob_selection) + "_K_" + str(self.n_bits) + "_Rcal_" + str(
            self.n_Rcal) + "_Net_" + str(self.network_type) + "_" + timestamp
        return id_tests


def main():
    # 1. parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("conf_file", type=str)
    args = parser.parse_args()

    # 2.0 create Setup
    setup_env = Setup_env(args.conf_file)
    id_tests = setup_env.id_tests()
    NTRANS = setup_env.n_bits
    NTRAIN = setup_env.n_train_offset
    global BATCH_SIZE
    BATCH_SIZE = setup_env.batch_size


    if setup_env.save_tests:
        setup_env.save()

    # 2.1 create Setup
    setup = Setup(args.conf_file)

    # 2.2 add observer
    observer = Observer(network_type=setup_env.network_type, 
                        dataset=setup_env.dataset, rgb_channels = setup_env.rgb_channels, 
                         height = setup_env.height, width = setup_env.width)
    setup.add_clients(observer)

    # 3. run N rounds OR load pre-trained models
    setup.run(federated_runs=NTRAIN)
    # setup.load("...")

    # 4. create Receiver
    receiver = Receiver(setup_env.n_channels,setup_env.frame_size,network_type = setup_env.network_type,
                         dataset=setup_env.dataset, rgb_channels = setup_env.rgb_channels, 
                         height = setup_env.height, width = setup_env.width)
    setup.add_clients(receiver)
    log_event('Receiver added')

    # 5. compute channel baseline
    # baseline = receiver.compute_baseline()
    while receiver.state != ReceiverState.Ready or receiver.frame_count != 0:
        setup.run(federated_runs=1)
        # pred = global_bias_prediction(setup.server, observer)
        # logging.info("SERVER: global prediction = %s", pred)

    logging.info("Attacker: ready to transmit with frame size %s", receiver.frame)

    # 6. create sender
    sender = Sender(receiver.images, receiver.labels, receiver.n_channels, receiver.frame, setup_env.pattern, network_type=setup_env.network_type,
                    dataset=setup_env.dataset, rgb_channels = setup_env.rgb_channels, 
                         height = setup_env.height, width = setup_env.width
                    )
    setup.add_clients(sender)
    log_event('Sender added')

    observer.set_frame(receiver.frame)
    observer.set_sample(create_samples(receiver.images, observer.dataset))

    # 7. perform channel calibration

    # 8. start transmitting
    successful_transmissions = 0
    error_rate = 0

    contingency_table_receiver = [[1,1],[1,1]]
    contingency_table_sender = [[0,0],[0,0]]
    logging.debug("Attacker: contingency table %s", contingency_table_receiver)

    for r in range(NTRANS):
        logging.info("Attacker: starting transmission frame")
        setup.run(federated_runs=receiver.frame)
        check, contingency_table_receiver, contingency_table_sender = check_transmission_success(sender, receiver, contingency_table_receiver, contingency_table_sender)
        successful_transmissions += check
        error_rate += (receiver.n_channels - check)
        logging.info("Attacker: SUCCESS rate = %s/%s", check, receiver.n_channels)

        log_event("Transmissions: " + str(r))
        log_event("Successful Transmissions: " + str(successful_transmissions))
        log_event("Errors:" + str(error_rate))

    logging.info("ATTACK TERMINATED: %s/%s bits succesfully transimitted", successful_transmissions, (NTRANS*receiver.n_channels))

    if(receiver.n_channels == 2):
        logging.info("CONTINGENCY TABLE RECEIVER %s", contingency_table_receiver)
        logging.info("CONTINGENCY TABLE SENDER %s", contingency_table_sender)
        p_value = chi_squared_test(contingency_table_receiver)
        logging.info("P-VALUE = %s", p_value)
        log_event("CONTINGENCY TABLE " +str(contingency_table_receiver)+ " P-VALUE = " + str(p_value))

    log_event("FINAL SUCCESSFUL TRANSMISSIONS: " + str(successful_transmissions) )
    log_event("FINAL ERROR: " + str(error_rate))
    log_event("ERROR RATE: " + str(error_rate))

    sdf = pandas.DataFrame(score_dict)
    # logging.info("CSV NAME: %s", os.path.join(setup_env.path, SCORE_LOG))
    sdf.to_csv(os.path.join(setup_env.path, SCORE_LOG))
    edf = pandas.DataFrame(event_dict)
    edf.to_csv(os.path.join(setup_env.path, EVENT_LOG))


def check_transmission_success(s, r, tr, ts):
    result = 0
    if s.bit is not None:
        if r.n_channels == 2:
            tr[r.bit[0]][r.bit[1]] += 1
            ts[s.bit[0]][s.bit[1]] += 1

        for c in range(len(s.bit)):
            if s.bit[c] == r.bit[c]:
                result += 1
            s.bit[c] = None
            r.bit[c] = None

    return result, tr, ts


if __name__ == '__main__':
    logging.basicConfig(format='[+] %(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
    main()
