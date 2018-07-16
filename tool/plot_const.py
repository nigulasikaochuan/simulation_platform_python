import matplotlib.pyplot as plt
from numpy import real, imag


def plot_const(signal):
    '''

    :param signal: Signal object
    :return: None
    '''
    plt.figure()
    plt.scatter(real(signal.data_sample[0, 1:10000]),imag(signal.data_sample[0,1:10000]),c='deepskyblue',marker='.')

    plt.show()


