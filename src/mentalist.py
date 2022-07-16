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


class ReaderState(enum.Enum):
    Crafting = 1
    Calibrating = 2
    Ready = 3
    Reading = 4

class Reader(Client):

    def __init__(self,n_classes,frame_size,network_type, dataset, rgb_channels, height, width):
        self.n_classes = n_classes
        self.images = ([None]*n_classes)*n_classes
        self.labels = ([None]*n_channels)*n_classes
        self.selection_count = 0
        self.frame = frame_size
        self.frame_count = 0
        self.state = ReaderState.Crafting
        x_train = numpy.random.random(size=[1,height,width,rgb_channels])
        y_train = numpy.random.random(size=[1,height,width,rgb_channels])
        x_train = x_train.astype('float32')
        super().__init__("Reader", x_train, y_train, x_train, y_train,network_type=network_type, dataset=dataset, rgb_channels=rgb_channels, height=height, width=width)

    def call_training(self, n_of_epoch):
        logging.debug("Reader: call_training()")

        if self.state == ReaderState.Calibrating:

            self.selection_count += 1
            logging.info("Reader: selected %s times", self.selection_count)
            if self.selection_count > NSELECTION:
                self.state = ReaderState.Ready
        else:
            pass

    # mindreading
    def update_model_weights(self, main_model):
        logging.debug("Reader: update_model_weights()")
        super().update_model_weights(main_model)

        logging.debug("Reader: frame_count = %s", self.frame_count)

        if self.state == ReaderState.Crafting:
            self.craft()
        elif self.state == ReaderState.Calibrating:
            self.calibrate()
        else:  # self.state == ReceiverState.Transmitting:
            self.read_from_model()

    def label_predict(self, x_pred):
        prediction = self.predict(x_pred)
        logging.debug("Reader: prediction %s", prediction)
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
            logging.info("Reader: frame starts with = %s", self.frame_start)
        elif self.frame_count == self.frame - 1:
            logging.info("Reader: frame ends with = %s", self.frame_end)
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

    # crafts a lattice of n(n-1)/2 edge samples
    def craft(self, n_classes):

        logging.info("Sender: crafting probe samples")

        for class_j in range(self.n_channel):
            for class_i in range(class_j-1):
                self.search_edge_example(class_i, class_j, size)

        logging.info("Reader: probes ready")

        if self.frame < 1:
            self.state = ReaderState.Calibrating
        else:
            self.state = ReaderState.Ready

    def search_edge_example(self, class_i, class_j, size):
        if self.dataset == 'mnist':
            self.search_edge_example_mnist(class_i, class_j, size)
        elif self.dataset == 'cifar':
            self.search_edge_example_cifar(class_i, class_j, size)


    def search_edge_example_cifar(self, class_i, class_j, size):
        for ii in range(size-1):
            image_i = bl.linearize(bl.get_image(ii))
            i_label = self.label_predict(create_sample(image_i, self.dataset))
            if i_label == class_i:
                for jj in range(size-1)
                image_j = bl.linearize(bl.get_image(jj))
                j_label = self.label_predict(create_sample(image_j, self.dataset))
                if j_label == class_j:
                    imageH = bl.hmix(image_i, image_j, ALPHA)
                    H_label = self.label_predict(create_sample(imageH, self.dataset))

                    alpha, y0_label, y1_label = self.hsearch(image_i, image_j, i_label, H_label, 0, ALPHA)

                    if alpha > 0:
                        logging.info("Reader: found hmix(%s, %s, %s) = %s | %s", i, j, alpha, y0_label, y1_label)
                        self.images[class_i][class_j] = bl.hmix(image_i, image_j, alpha)
                        self.labels[class_i][class_j] = [y0_label.cpu(), y1_label.cpu()]
                        return
                    else:
                        imageV = bl.vmix(image_i, image_j, ALPHA)
                        V_label = self.label_predict(create_sample(imageV, self.dataset))

                        alpha, y0_label, y1_label = self.vsearch(image_i, image_j, i_label, V_label, 0, ALPHA)

                        if alpha > 0: # and not y0_label in allocated and not y1_label in allocated:
                            logging.info("Reader: found vmix(%s, %s, %s) = %s | %s", i, j, alpha, y0_label, y1_label)
                            self.images[class_i][class_j] = bl.vmix(image_i, image_j, alpha)
                            self.labels[class_i][class_j] = [y0_label.cpu(), y1_label.cpu()]
                            return
        logging                    

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

# MOD BELOW

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
    observer = Observer()
    setup.add_clients(observer)

    # 3. run N rounds OR load pre-trained models
    setup.run(federated_runs=NTRAIN)
    # setup.load("...")

    # 4. create Reader
    reader = Reader()
    setup.add_clients(reader)
    log_event('Reader added')

    # 5. compute lattice
    while reader.state != ReaderState.Ready or reader.frame_count != 0:
        setup.run(federated_runs=1)
        # pred = global_bias_prediction(setup.server, observer)
        # logging.info("SERVER: global prediction = %s", pred)

    logging.info("Attacker: ready to operate with frame size %s", receiver.frame)

    observer.set_frame(receiver.frame)
    observer.set_sample(create_samples(reader.images, observer.dataset))

    for r in range(NTRANS):
        logging.info("Attacker: starting reading frame")
        setup.run(federated_runs=receiver.frame)
        ## MUST READ THE SAMPLE SCORES

        log_event("Reading: ")

        ## MUST SAVE RESULTS

    logging.info("ATTACK TERMINATED: ")

    log_event("FINAL RESULT: ")

    sdf = pandas.DataFrame(score_dict)
    # logging.info("CSV NAME: %s", os.path.join(setup_env.path, SCORE_LOG))
    sdf.to_csv(os.path.join(setup_env.path, SCORE_LOG))
    edf = pandas.DataFrame(event_dict)
    edf.to_csv(os.path.join(setup_env.path, EVENT_LOG))



if __name__ == '__main__':
    logging.basicConfig(format='[+] %(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
    main()
