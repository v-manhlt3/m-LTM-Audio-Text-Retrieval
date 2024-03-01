from tools.dataset import pack_dataset_to_hdf5
from loguru import logger

if __name__ == '__main__':

    logger.info('Packing dataset to hdf5 files.')
    logger.info('Packing AudioCaps...')
    pack_dataset_to_hdf5('AudioCaps')
    logger.info('AudioCaps done!')
    logger.info('Packing Clotho...')
    pack_dataset_to_hdf5('Clotho')
    logger.info('Clotho done!')
