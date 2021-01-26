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
from baseliner import cancelFromLeft

from datetime import datetime
import yaml
import os
import subprocess


# Just a 8 (n. 404 in MNIST)
ORIGINAL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 5, 31, 130, 222, 255, 255, 154, 86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            2, 11, 101, 253, 244, 241, 241, 244, 253, 213, 136, 84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 19, 92,
            145, 200, 19, 111, 33, 0, 0, 33, 217, 253, 253, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 242, 241,
            81, 55, 5, 8, 0, 0, 0, 0, 81, 253, 241, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 253, 253, 191,
            172, 95, 16, 0, 0, 0, 97, 253, 118, 135, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 93, 168, 221, 253,
            235, 101, 38, 9, 66, 236, 253, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 108, 218, 150,
            253, 188, 174, 239, 167, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 119, 253, 253,
            253, 228, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 253, 253, 253, 253, 185,
            8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 50, 139, 158, 191, 191, 241, 253, 196, 114,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 147, 221, 149, 30, 0, 0, 0, 89, 242, 253, 154, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 205, 253, 173, 0, 0, 0, 0, 0, 0, 114, 253, 154, 120, 14, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 126, 253, 183, 17, 0, 0, 0, 0, 0, 0, 17, 224, 185, 237, 18, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 16, 236, 227, 16, 0, 0, 0, 0, 0, 0, 0, 0, 217, 253, 253, 122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 65, 253, 176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 171, 253, 253, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 143,
            253, 220, 37, 0, 0, 0, 0, 0, 0, 0, 0, 142, 253, 253, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 92, 253, 253,
            220, 88, 0, 0, 0, 0, 0, 34, 209, 250, 253, 157, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 102, 185, 158,
            250, 150, 112, 112, 112, 148, 241, 253, 212, 196, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 20,
            155, 253, 253, 253, 253, 253, 216, 135, 25, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 30,
            129, 129, 129, 129, 149, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

SEARCH_THREASHOLD = 1 / (28 * 28)

NTRAIN = 1  # rounds of training
NTRANS = 10  # rounds for transmission tests
DELTA = 0.1
ALPHA = 0.42
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

error_rate = 0

save_path = ""


def increase_error_rate(error_rate):
    error_rate += 1


def log_score(x, y):
    score_dict['X'].append(x)
    score_dict['Y'].append(y)


def log_event(x, e):
    event_dict['X'].append(x)
    event_dict['E'].append(e)


hl, = plt.plot([], [])
plt.ylim([20, 55])
plt.xlim([0, NTRAIN + (NTRANS * 12)])

plt.xlabel('Time (FL rounds)', fontdict=font)
plt.ylabel('Prediction', fontdict=font)
plt.title('Covert Channel Comm. via Label Attack to a FL model', fontdict=font)


def update_plot(x, y):
    hl.set_xdata(numpy.append(hl.get_xdata(), [x]))
    hl.set_ydata(numpy.append(hl.get_ydata(), [y]))


def add_vline(xv):
    plt.axvline(x=xv)


def signal_handler(sig, frame):
    save_stats()
    sys.exit(0)


def save_stats():
    y_values = hl.get_ydata()
    y_min = min(y_values) - DELTA_PLT_Y
    y_max = max(y_values) + DELTA_PLT_Y
    plt.ylim(y_min, y_max)
    x_values = hl.get_xdata()
    x_min = min(x_values) - DELTA_PLT_X
    x_max = max(x_values) + DELTA_PLT_X
    plt.xlim(x_min, x_max)
    plt.savefig(save_path + 'output.png', dpi=300)
    plt.savefig(save_path +'output.svg', dpi=300)
    sdf = pandas.DataFrame(score_dict)
    sdf.to_csv(save_path + SCORE_LOG)
    edf = pandas.DataFrame(event_dict)
    edf.to_csv(save_path + EVENT_LOG)


# compute slope through least square method
def slope(y):
    numer = 0
    denom = 0
    mean_x = (len(y) - 1) / 2
    mean_y = numpy.mean(y)
    for x in range(len(y)):
        numer += (x - mean_x) * (y[x] - mean_y)
        denom += (x - mean_x) ** 2
    m = numer / denom
    return m


signal.signal(signal.SIGINT, signal_handler)


def create_sample(image):
    x_train = numpy.array([image])
    x_train = x_train.astype('float32')
    x_train /= 255
    # return x_train[[0]]
    return torch.from_numpy(x_train[[0]])


class ReceiverState(enum.Enum):
    Crafting = 1
    Calibrating = 2
    Ready = 3
    Transmitting = 4


class Sender(Client):

    def __init__(self, receiverImage, y_train, frame):
        self.bit = None
        self.sent = False
        self.frame_count = -1
        self.frame = frame
        self.frame_start = None
        x_train = numpy.array([receiverImage, receiverImage])
        x_train = x_train.astype('float32')
        x_train /= 255
        super().__init__("Sender", x_train, y_train, x_train, y_train)

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
            self.frame_start = self.label_predict(self.x_train[[0]])
            logging.info("Sender: frame starts with %s", self.frame_start)
            self.bit = random.randint(0, 1)
            logging.info("Sender: SENDING %s", self.bit)

        self.frame_count = (self.frame_count + 1) % self.frame

    def label_predict(self, x_pred):
        prediction = self.predict(x_pred)
        logging.debug("Sender: prediction %s", prediction)
        # TODO: must return max element only
        return torch.argmax(prediction)

    # forces biases to transmit one bit through the model
    def send_to_model(self, n_of_epoch):

        if self.bit == 1:
            # change prediction
            logging.info("Sender: injecting bias 1")

            if self.frame_start == self.y_train[[0]]:
                y_train_trans = self.y_train[1:2]
            else:
                y_train_trans = self.y_train[0:1]

            logging.info("Sender: index %s", y_train_trans)
            # bias injection dataset
            train_ds = TensorDataset(self.x_train[0:1], y_train_trans)
            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)

            # bias testing dataset
            test_ds = TensorDataset(self.x_train[0:1], y_train_trans)
            test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

            for epoch in range(n_of_epoch):
                train_loss, train_accuracy = self.train(train_dl)
                test_loss, test_accuracy = self.validation(test_dl)

        else:

            logging.info("Sender: injecting bias 0")
            # do nothing, prediction should stay unchanged


class Receiver(Client):

    def __init__(self, oImage):
        self.bit = None
        self.original = oImage
        self.image = None
        self.selection_count = 0
        self.frame = 0
        self.frame_count = 0
        self.frame_start = 0
        self.frame_end = 0
        self.state = ReceiverState.Crafting
        x_train = numpy.array([])
        y_train = numpy.array([])
        x_train = x_train.astype('float32')
        super().__init__("Receiver", x_train, y_train, x_train, y_train)

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
        # TODO: must return max element only
        return torch.argmax(prediction)

    def read_from_model(self):

        x_pred = torch.from_numpy(self.x_train[[0]])
        pred = self.label_predict(x_pred)

        if self.frame_count == 0:
            self.frame_start = pred
            logging.info("Receiver: frame starts with = %s", pred)
        elif self.frame_count == self.frame - 1:
            self.frame_end = pred
            logging.info("Receiver: frame ends with = %s", pred)

            if self.frame_start == self.frame_end:
                self.bit = 0
            else:
                self.bit = 1
            logging.info("Receiver: RECEIVED: %s", self.bit)
        else:
            pass

        self.frame_count = (self.frame_count + 1) % self.frame

    def calibrate(self):
        self.frame += 1

    def craft(self):

        xB_sample = create_sample(self.original)
        yB_label = self.label_predict(xB_sample)

        imageT = cancelFromLeft(self.original, 0.5)
        xT_sample = create_sample(imageT)
        yT_label = self.label_predict(xT_sample)

        alpha, y0_label, y1_label = self.search(yB_label, yT_label, 0, ALPHA)

        logging.info("Receiver: found edge cut alpha = %s, y0 = %s, y1 = %s", alpha, y0_label, y1_label)

        self.image = cancelFromLeft(self.original, alpha)
        self.x_train = numpy.array([self.image, self.image])
        self.y_train = numpy.array([y0_label, y1_label])
        self.x_train = self.x_train.astype('float32')
        self.x_train /= 255

        self.state = ReceiverState.Calibrating

    def search(self, y0_label, y1_label, alpha_min, alpha_max):

        logging.debug("Searching between %s and %s", y0_label, y1_label)
        assert (y0_label != y1_label), "Labels cannot be equal"

        if alpha_max < alpha_min + SEARCH_THREASHOLD:
            return alpha_min, y0_label, y1_label

        imageM = cancelFromLeft(self.original, (alpha_min + alpha_max) / 2)
        xM_sample = create_sample(imageM)
        yM_label = self.label_predict(xM_sample)
        if y0_label != yM_label:
            return self.search(y0_label, yM_label, alpha_min, (alpha_min + alpha_max) / 2)
        else:
            return self.search(yM_label, y1_label, (alpha_min + alpha_max) / 2, alpha_max)


class Observer(Client):

    def __init__(self):
        self.frame_count = 0
        self.frame = 0
        self.x = 0
        self.sample = None
        x_train = numpy.array([])
        y_train = numpy.array([])
        x_train = x_train.astype('float32')
        super().__init__("Observer", x_train, y_train, x_train, y_train)

    # Covert channel send
    def call_training(self, n_of_epoch):
        pass

    def set_frame(self, frame):
        self.frame = frame

    def set_sample(self, s):
        self.sample = s

    def update_model_weights(self, main_model):
        logging.debug("Observer: update_model_weights()")
        super().update_model_weights(main_model)

        if self.sample != None:
            pred = self.predict(self.sample)
            logging.debug("Observer: global prediction = %s, frame_count = %s", pred, self.frame_count)
            update_plot(self.x, torch.argmax(pred))
            log_score(self.x, pred)

        if self.frame > 0:
            if self.frame_count == 0:
                add_vline(self.x)
                log_event(self.x, 'Frame start')
            self.frame_count = (self.frame_count + 1) % self.frame

        self.x += 1


class Setup_env:
    '''Setup simulation environment from YAML configuration file.
    '''

    def __init__(self, conf_file):
        self.conf_file = conf_file

        self.settings = self.load(self.conf_file)

        self.save_tests = self.settings['setup']['save_tests']
        self.saving_tests_dir = self.settings['setup']['tests_dir']
        self.prob_selection = self.settings['setup']['prob_sel']
        self.n_bits = self.settings['setup']['n_bits']
        self.n_Rcal = self.settings['setup']['n_Rcal']
        self.saved = False

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
        id_folder = subprocess.check_output('cat /proc/self/cgroup | grep "docker" | sed  s/\\\\//\\\\n/g | tail -1', shell=True).decode("utf-8").rstrip()
        timestamp = self.start_time.strftime("%Y%m%d%H%M%S")
        self.path = os.path.join(self.saving_tests_dir, id_folder)
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
            self.n_Rcal) + "_" + timestamp
        return id_tests


def main():
    # 1. parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("conf_file", type=str)
    args = parser.parse_args()

    # 2.0 create Setup
    setup_env = Setup_env(args.conf_file)
    id_tests = setup_env.id_tests()

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

    # 4. create Receiver
    receiver = Receiver(ORIGINAL)
    setup.add_clients(receiver)
    log_event(observer.x, 'Receiver added')

    # 5. compute channel baseline
    # baseline = receiver.compute_baseline()
    while receiver.state != ReceiverState.Ready or receiver.frame_count != 0:
        setup.run(federated_runs=1)
        # pred = global_bias_prediction(setup.server, observer)
        # logging.info("SERVER: global prediction = %s", pred)

    logging.info("Attacker: ready to transmit with frame size %s", receiver.frame)

    # 6. create sender
    sender = Sender(receiver.image, receiver.y_train, receiver.frame)
    setup.add_clients(sender)
    log_event(observer.x, 'Sender added')
    observer.set_frame(receiver.frame)

    observer.set_sample(create_sample(receiver.image))

    # 7. perform channel calibration

    # 8. start transmitting
    successful_transmissions = 0
    for r in range(NTRANS):
        logging.info("Attacker: starting transmission frame")
        setup.run(federated_runs=receiver.frame)
        successful_transmissions += check_transmission_success(sender, receiver)
        log_event(observer.x, "Transmissions: " + str(successful_transmissions))

    logging.info("ATTACK TERMINATED: %s/%s bits succesfully transimitted", successful_transmissions, NTRANS)

    save_stats()

    log_event(observer.x, "ERROR RATE: " + str(error_rate))

    save_stats()

    y_values = hl.get_ydata()
    y_min = min(y_values) - DELTA_PLT_Y
    y_max = max(y_values) + DELTA_PLT_Y
    plt.ylim(y_min, y_max)
    x_values = hl.get_xdata()
    x_min = min(x_values) - DELTA_PLT_X
    x_max = max(x_values) + DELTA_PLT_X
    plt.xlim(x_min, x_max)
    # logging.info("FIGURE NAME: %s", os.path.join(setup_env.path, id_tests + '.png'))
    plt.savefig(os.path.join(setup_env.path, id_tests + '.png'), dpi=300)
    plt.savefig(os.path.join(setup_env.path, id_tests + '.svg'), dpi=300)

    sdf = pandas.DataFrame(score_dict)
    # logging.info("CSV NAME: %s", os.path.join(setup_env.path, SCORE_LOG))
    sdf.to_csv(os.path.join(setup_env.path, SCORE_LOG))
    edf = pandas.DataFrame(event_dict)
    edf.to_csv(os.path.join(setup_env.path, EVENT_LOG))


def check_transmission_success(s, r):
    result = 0
    if s.bit != None:
        if s.bit == r.bit:
            logging.info("Attacker: transmission SUCCESS")
            result = 1
        else:
            logging.info("Attacker: transmission FAIL")
            increase_error_rate(error_rate)
        s.bit = None
        r.bit = None
    return result


if __name__ == '__main__':
    logging.basicConfig(format='[+] %(levelname)s: %(message)s', level=logging.INFO)
    main()
