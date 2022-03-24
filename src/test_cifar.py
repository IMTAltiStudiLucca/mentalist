import os
from re import I 
import sys 
import shutil
import argparse



def main(SIMULATION_NUMBER):
    DIR_FILE_SIMULATIONS = 'simulations'
    FILE_NAME_PREFIX = 'simulazione_'
    FILE_NAME_SUFFIX = '.yaml'
    SIMULATION_FILE_NAME = FILE_NAME_PREFIX+SIMULATION_NUMBER+FILE_NAME_SUFFIX 
    SCRIPTS_TO_BE_COPIED = ['federated_learning.py','label-attack-multichannel.py','jpg_baseliner.py','baseliner.py','chi_squared_test.py']
    FOLDERS_TO_BE_COPIED = ['morphomnist','runs']
    NUM_OF_EXECUTIONS = 5
    SIMULATION_SETUP = os.path.join(DIR_FILE_SIMULATIONS,SIMULATION_FILE_NAME)

    for i in range(NUM_OF_EXECUTIONS):
        test_folder = os.path.join(SIMULATION_NUMBER,str(i))
        if not os.path.isdir(SIMULATION_NUMBER):
            print('CREATE SIMULATION FOLDER {}'.format(SIMULATION_NUMBER))
            os.mkdir(SIMULATION_NUMBER)
        if not os.path.isdir(test_folder):
            print('\tCREATE TEST {} DIRECTORY {}'.format(i,test_folder))    
            os.mkdir(test_folder)
        for sc in SCRIPTS_TO_BE_COPIED:
            shutil.copy2(sc,test_folder)
        for fl in FOLDERS_TO_BE_COPIED:
            dest_folder = os.path.join(test_folder,fl)
            shutil.copytree(fl,dest_folder,dirs_exist_ok=True)
        yaml_path = os.path.join(DIR_FILE_SIMULATIONS,SIMULATION_FILE_NAME)
        shutil.copy2(yaml_path,test_folder)
        print(os.listdir(test_folder))
        root_dir = os.getcwd()
        os.chdir(test_folder)
        command = 'python3.8 label-attack-multichannel.py {} >> "simulazione_{}.out" 2>&1 '.format(SIMULATION_FILE_NAME,i)
        print(command)
        os.system(command)
        os.chdir(root_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("SIMULATION_NUMBER", type=str)
    args = parser.parse_args()
    SIMULATION_NUMBER = args.SIMULATION_NUMBER
    main(SIMULATION_NUMBER)