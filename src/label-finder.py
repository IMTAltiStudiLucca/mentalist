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
import random

from datetime import datetime
import yaml
import os
import subprocess

SEARCH_THREASHOLD = 1 / (28 * 28)
MNIST_SIZE = 60000
SAMPLES = 600
ALPHA = 0.5


def create_sample(image):
    x_train = numpy.array([image])
    x_train = x_train.astype('float32')
    x_train /= 255
    # return x_train[[0]]
    return torch.from_numpy(x_train[[0]])

class Finder(Client):

    def __init__(self, network_type):
        random.seed()
        self.i = random.randint(0,MNIST_SIZE)
        self.j = random.randint(0,MNIST_SIZE)
        self.image_i = None
        self.image_j = None
        x_train = numpy.array([])
        y_train = numpy.array([])
        x_train = x_train.astype('float32')
        super().__init__("Finder", x_train, y_train, x_train, y_train,network_type=network_type)

    def call_training(self, n_of_epoch):
        pass

    # Covert channel receive
    def update_model_weights(self, main_model):
        logging.debug("Finder: update_model_weights()")
        super().update_model_weights(main_model)

        logging.info("Finder: starting edge sample search at %s, %s", self.i, self.j)

        i = random.sample(range(MNIST_SIZE-1), SAMPLES)
        j = random.sample(range(MNIST_SIZE-1), SAMPLES)

        fh,fv = 0,0
        for x in range(SAMPLES):
            print("Processing: {:.2%}".format(x/SAMPLES)) #, end="\r", flush=True)
            ih,iv = self.craft(i[x],j[x])
            fh += ih
            fv += iv

        print("Edge examples search terminarated")
        print("Total attempts " + str(SAMPLES * 2) + " H " + str(SAMPLES) + " V " + str(SAMPLES))
        print("Found edge examples " + str(fh+fv) + " H " + str(fh) + " V " + str(fv))
        logging.info("Finder: edge sample search terminated")


    def label_predict(self, x_pred):
        prediction = self.predict(x_pred)
        logging.debug("Finder: prediction %s", prediction)
        # TODO: must return max element only
        return torch.argmax(prediction)

    def craft(self, x1, x2):

        self.i = x1
        self.j = x2

        fh,fv = 0,0

        self.image_i = bl.linearize(bl.get_image(self.i))
        self.image_j = bl.linearize(bl.get_image(self.j))
        i_label = self.label_predict(create_sample(self.image_i))

        imageH = bl.hmix(self.image_i, self.image_j, ALPHA)
        H_label = self.label_predict(create_sample(imageH))

        alpha, y0_label, y1_label = self.hsearch(i_label, H_label, 0, ALPHA)

        if alpha > 0:
            logging.info("Finder: found hmix(%s, %s, %s) = %s | %s", alpha, self.i, self.j, y0_label, y1_label)
            fh += 1
        else:
            logging.debug("Finder: not found for (%s,%s)", self.i, self.j)

        imageV = bl.vmix(self.image_i, self.image_j, ALPHA)
        V_label = self.label_predict(create_sample(imageV))

        alpha, y0_label, y1_label = self.vsearch(i_label, V_label, 0, ALPHA)

        if alpha > 0:
            logging.info("Finder: found vmix(%s, %s, %s) = %s | %s", alpha, self.i, self.j, y0_label, y1_label)
            fv += 1
        else:
            logging.debug("Finder: not found for (%s,%s)", self.i, self.j)

        return fh, fv

        #self.i += 1

        #if self.i > MNIST_SIZE:
        #    self.i = 0
        #    self.j = (self.j + 1) % MNIST_SIZE

    def hsearch(self, y0_label, y1_label, alpha_min, alpha_max):

        logging.debug("H-searching between %s and %s", y0_label, y1_label)

        if y0_label == y1_label:
            return -1,None,None

        if alpha_max < alpha_min + SEARCH_THREASHOLD:
            return alpha_min, y0_label, y1_label

        imageM = bl.hmix(self.image_i, self.image_j, (alpha_min + alpha_max) / 2)
        yM_label = self.label_predict(create_sample(imageM))
        if y0_label != yM_label:
            return self.hsearch(y0_label, yM_label, alpha_min, (alpha_min + alpha_max) / 2)
        else:
            return self.hsearch(yM_label, y1_label, (alpha_min + alpha_max) / 2, alpha_max)

    def vsearch(self, y0_label, y1_label, alpha_min, alpha_max):

        logging.debug("V-searching between %s and %s", y0_label, y1_label)

        if y0_label == y1_label:
            return -1,None,None

        if alpha_max < alpha_min + SEARCH_THREASHOLD:
            return alpha_min, y0_label, y1_label

        imageM = bl.vmix(self.image_i, self.image_j, (alpha_min + alpha_max) / 2)
        yM_label = self.label_predict(create_sample(imageM))
        if y0_label != yM_label:
            return self.vsearch(y0_label, yM_label, alpha_min, (alpha_min + alpha_max) / 2)
        else:
            return self.vsearch(yM_label, y1_label, (alpha_min + alpha_max) / 2, alpha_max)

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

    # 2.2 add Finder
    finder = Finder(network_type=setup_env.network_type)
    setup.add_clients(finder)

    # 3. run N rounds OR load pre-trained models
    # setup.run(federated_runs=1)
    # setup.load("...")

    logging.info("TERMINATED")


if __name__ == '__main__':
    logging.basicConfig(format='[+] %(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
    main()
