import subprocess
import tarfile
import numpy as np
import h5py
import csv
import imageio
import os
import random
import glob
from math import ceil


class Dataset:
    def __init__(self, batch_size=1000, download=False, gen_images=False, gen_hdf5=False):
        if download:
            self.download_kaggle_dataset()
        if gen_images:
            self.convert_csv_to_images()
        if gen_hdf5:
            self.generate_hdf5_file_from_images()

        hdf5_path = "./dataset/dataset.hdf5"
        self.hdf5_file = h5py.File(hdf5_path, "r")
        self.batch_size = batch_size
        self.current_batch = 0
        self.data_length = self.hdf5_file["train_img"].shape[0]
        self.max_batch = int(ceil(self.data_length / self.batch_size))

    def close(self):
        self.hdf5_file.close()

    def download_kaggle_dataset(self):
        """
        Call the kaggle API and download the Kaggle dataset used in the  challenge referenced
        in the documentation.
        """
        subprocess.check_output(
            "kaggle competitions download -c challenges-in-representation-learning-facial-expression-recognition-challenge".split(
                " "))

        # Extract the kaggle dataset
        data_tar = "fer2013.tar.gz"
        tar = tarfile.open(data_tar)
        tar.extractall()
        tar.close()

    def __gen_chunks(self, reader, chunksize=20):
        """
        Chunk generator.
        Take a CSV reader and yield chunksize sized slices.
        Parameters:
        reader (csv.reader): CSV reader of a dataset
        chunksize (int): Size of each chunk
        """
        chunk = []
        for i, line in enumerate(reader):
            if i % chunksize == 0 and i > 0:
                yield chunk
                del chunk[:]
            chunk.append(line)
        yield chunk

    def convert_csv_to_images(self):
        """Read the previously downloaded dataset and generates a set images from it"""

        reader = csv.reader(open('fer2013/fer2013.csv', 'r'))
        reader.__next__()  # Skip the dataset header

        # Create folders if they don't exist
        for i in range(7):
            filename = "./dataset/%d/" % i
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        indexs = np.full((7), 0)
        for chunk in self.__gen_chunks(reader):
            for row in chunk:
                folder = "dataset/%s/" % (row[0])
                img = np.fromstring(row[1], dtype=np.uint8, sep=' ').reshape(48, 48)
                imageio.imwrite(folder + str(indexs[int(row[0])]) + ".jpg", img)
                indexs[int(row[0])] += 1

    def generate_hdf5_file_from_images(self):
        """
        Take all images from a dataset folder and generate an hdf5 file from them
        Reference: http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
        """
        shuffle_data = True  # shuffle the addresses before saving
        hdf5_path = './dataset/dataset.hdf5'  # address to where you want to save the hdf5 file
        dataset_path = 'dataset/*/*.jpg'
        # read addresses and labels from the 'train' folder
        addrs = glob.glob(dataset_path)
        labels = [int(x[8]) for x in addrs]

        # shuffle data
        if shuffle_data:
            random.seed(0)
            c = list(zip(addrs, labels))
            random.shuffle(c)
            addrs, labels = zip(*c)

        # Divide the data into 60% train, 20% validation, and 20% test
        train_addrs = addrs[0:int(0.6 * len(addrs))]
        train_labels = labels[0:int(0.6 * len(labels))]
        val_addrs = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
        val_labels = labels[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
        test_addrs = addrs[int(0.8 * len(addrs)):]
        test_labels = labels[int(0.8 * len(labels)):]

        # Get each subdataset sizes
        train_shape = (len(train_addrs), 48, 48)
        val_shape = (len(val_addrs), 48, 48)
        test_shape = (len(test_addrs), 48, 48)

        # Create hdf5 file
        hdf5_file = h5py.File(hdf5_path, mode="w")

        hdf5_file.create_dataset("train_img", train_shape, np.uint8)
        hdf5_file.create_dataset("val_img", val_shape, np.uint8)
        hdf5_file.create_dataset("test_img", test_shape, np.uint8)

        # Add labels to the hdf5 file
        hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.uint8)
        hdf5_file["train_labels"][...] = train_labels
        hdf5_file.create_dataset("val_labels", (len(val_addrs),), np.uint8)
        hdf5_file["val_labels"][...] = val_labels
        hdf5_file.create_dataset("test_labels", (len(test_addrs),), np.uint8)
        hdf5_file["test_labels"][...] = test_labels

        # Read each image and save its data in the hdf5 file
        for i in range(len(train_addrs)):
            # print how many images are saved every 1000 images
            if i % 1000 == 0 and i > 1:
                print('Train data: {}/{}'.format(i, len(train_addrs)))

            addr = train_addrs[i]
            img = imageio.imread(addr)

            hdf5_file["train_img"][i, ...] = img

        for i in range(len(val_addrs)):
            # print how many images are saved every 1000 images
            if i % 1000 == 0 and i > 1:
                print('Validation data: {}/{}'.format(i, len(val_addrs)))

            addr = val_addrs[i]
            img = imageio.imread(addr)
            hdf5_file["val_img"][i, ...] = img

        for i in range(len(test_addrs)):
            # print how many images are saved every 1000 images
            if i % 1000 == 0 and i > 1:
                print('Test data: {}/{}'.format(i, len(test_addrs)))

            addr = train_addrs[i]
            img = imageio.imread(addr)
            hdf5_file["test_img"][i, ...] = img

        hdf5_file.close()

    def get_next_batch(self):
        """
        Obtain the next available batch in the dataset.
        Throws a BufferError if there aren't more batches to obtain.
        Returns:
        numpy.ndarray: Batch of images from the dataset
        numpy.ndarray: Batch of the respective labels for the images
        """
        if self.current_batch == self.max_batch:
            raise BufferError("No more batches remain in the dataset")
        start = self.current_batch * self.batch_size  # index of the first image in this batch
        end = min(
            [(self.current_batch + 1) * self.batch_size, self.data_length])  # index of the last image in this batch

        images = self.hdf5_file["train_img"][start:end, ...]
        labels = self.hdf5_file["train_labels"][start:end]

        self.current_batch += 1

        return images, labels

    def is_empty(self):
        """
        Check if the last batch has been obtained

        Returns:
        bool: True if no more batches can be obtained. False otherwise.
        """

        return self.current_batch == self.max_batch

    def reset_current_batch(self):
        """
        Reset the current batch to 0 again without creating a new class.
        """
        self.current_batch = 0

    def gen_batches(self):
        """
        Batch generator to be use in for loops.
        Returns:
        numpy.ndarray: Batch of images from the dataset
        numpy.ndarray: Batch of the respective labels for the images
        """
        batches_list = [x for x in range(self.max_batch)]
        for i in batches_list:
            i_s = i * self.batch_size  # index of the first image in this batch
            i_e = min([(i + 1) * self.batch_size, self.data_length])  # index of the last image in this batch

            images = self.hdf5_file["train_img"][i_s:i_e, ...]
            labels = self.hdf5_file["train_labels"][i_s:i_e]
            yield images, labels
