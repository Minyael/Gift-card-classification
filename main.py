"""
    Main file for the project.  
    
    This file is used to run the project and test the CNN.

"""
import sys
from src.webcam import webcam
from src.training import transfer_classification
from src.data_augmentation import data_augmentation
from src.image_download import download

def run(program_to_run):
    """
        Run the project.
    """
    if program_to_run == 'webcam':
        webcam()
    elif program_to_run == 'train':
        transfer_classification()
    elif program_to_run == 'augmentation':
        data_augmentation()
    elif program_to_run == 'download':
        download()

if __name__=='__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['webcam', 'train', 'augmentation', 'download']:
        run(sys.argv[1])
    else:
        print("Invalid command. Please use one of the following commands:")
        print("webcam, train, augmentation, download")
