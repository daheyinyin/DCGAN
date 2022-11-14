"""verify by svm"""
import time
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler


def verify_cifar10():
    """
        verify on cifar10 dataset
    """
    label_path_train = "./preprocess_Result/cifar10_label_ids_train.npy"
    label_path_test = "./preprocess_Result/cifar10_label_ids_test.npy"
    label_set_train = np.load(label_path_train, allow_pickle=True)
    label_set_test = np.load(label_path_test, allow_pickle=True)

    result_set_train = []
    for i in range(0, 500):
        result_file_train = './result_Files_train/dcgan_data_bs100_' + str(i) + "_train_0.bin"
        result = np.fromfile(result_file_train, dtype=np.float32).reshape(-1, 14336)
        result_set_train.append(result)

    result_set_train = np.array(result_set_train)
    result_set_train = result_set_train.reshape(-1, 14336)
    label_set_train = label_set_train.reshape(-1, 1)
    label_set_train = label_set_train.flatten()

    result_set_test = []
    for i in range(0, 100):
        result_file_test = './result_Files_test/dcgan_data_bs100_' + str(i) + "_test_0.bin"
        result = np.fromfile(result_file_test, dtype=np.float32).reshape(-1, 14336)
        result_set_test.append(result)
    result_set_test = np.array(result_set_test)
    result_set_test = result_set_test.reshape(-1, 14336)
    label_set_test = label_set_test.reshape(-1, 1)
    label_set_test = label_set_test.flatten()

    print("result_set_train.shape: ", result_set_train.shape)
    print("label_set_train.shape: ", label_set_train.shape)

    print("result_set_test.shape: ", result_set_test.shape)
    print("label_set_test.shape: ", label_set_test.shape)

    print("============================standradScaler")
    standardScaler = StandardScaler()
    standardScaler.fit(result_set_train)
    result_set_train_standard = standardScaler.transform(result_set_train)
    standardScaler.fit(result_set_test)
    result_set_test_standard = standardScaler.transform(result_set_test)

    print("============================training")
    clf = svm.SVC(max_iter=-1)
    start = time.time()
    print("result_set_train.shape: ", result_set_train_standard.shape)
    print("label_set_train.shape: ", label_set_train.shape)
    clf.fit(result_set_train_standard, label_set_train)
    t = time.time() - start
    print("train time:", t)

    print("============================testing")
    # Test on Training data
    print("result_set_test.shape: ", result_set_test_standard.shape)
    print("label_set_test.shape: ", label_set_test.shape)
    test_result = clf.predict(result_set_test_standard)
    accuracy = sum(test_result == label_set_test) / label_set_test.shape[0]
    print('Test accuracy: ', accuracy)


if __name__ == '__main__':
    print("============================verify_cifar10")
    verify_cifar10()
