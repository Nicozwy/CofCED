# coding: utf-8
import numpy
import torch
import random
def set_random_seed(seed):
    """ set random seed for numpy and torch, more information here:
        https://pytorch.org/docs/stable/notes/randomness.html
    Args:
        seed: the random seed to set
    """
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(name='auto'):
    """ choose device

    Returns:
        the device specified by name, if name is None, proper device will be returned

    """
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def calc_f1(tp, fp, fn, print_result=True):
    """ calculating f1

    Args:
        tp: true positive
        fp: false positive
        fn: false negative
        print_result: whether to print result

    Returns:
        precision, recall, f1

    """
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    if print_result:
        print(" precision = %f, recall = %f, micro_f1 = %f\n" % (precision, recall, f1))
    return precision, recall, f1


def main():
    pass


if __name__ == '__main__':
    main()
