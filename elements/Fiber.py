from common_package import *
from Signal.Signal import Signal
from Signal.Signal import WaveShape
from typing import List
import arrayfire as af

from elements.EDFA import EDFA


class Span(object):
    '''
        a span of fiber,the parameter of span should be supplied
        :param alpha ---> unit db/km
        :param beta2 ---> in s**2/km
        :param beta3 --->
        :param gamma ----> in 1/(w*km)
        :param length --->km
        :param is_cuda ---> whether use cuda, and now it must be True

        this function implements the split-fourier method, include GVD(also named CD), nonlinear

        distortation.
    '''

    def __init__(self, alpha, beta2, beta3, gamma, length, is_cuda=True):
        self.alpha = alpha
        self.beta2 = beta2
        self.beta3 = beta3
        self.gamma = gamma
        self.length = length
        self.is_cuda = is_cuda

    @property
    def alphalin(self):
        '''
            convert db/km to 1/km
        :return: 1/km
        '''
        return np.log(10 ** (self.alpha / 10))

    def prop(self, signal, step, mode='split-fourier'):
        '''
            main function to be called
        :param signal:
        :param step:
        :param mode:
        :return:
        '''
        if mode == 'split-fourier':
            self.split_fourier(signal, step)

    def split_fourier(self, signal, step):
        if self.is_cuda:
            self._split_fourier_cuda(signal, step)

    def _split_fourier_cuda(self, signal: Signal, step):
        '''
            This function is called by split_fourier,and should not be used outside

        :param signal: signal to traverse the span
        :param step: the step of split fourier
        :return: None
        '''
        af.set_backend('cuda')

        freq = fftfreq(len(signal.data_sample[0, :]), (signal.sps * signal.symbol_rate_in_hz) ** (-1))

        freq = af.Array(freq.ctypes.data, freq.shape, freq.dtype.char)

        signal_x = np.asarray(signal.data_sample[0, :])
        signal_y = np.asarray(signal.data_sample[1, :])

        signal_x = af.Array(signal_x.ctypes.data, signal_x.shape, dtype=signal_x.dtype.char)
        signal_y = af.Array(signal_y.ctypes.data, signal_x.shape, dtype=signal_y.dtype.char)

        Disper = (1j / 2) * self.beta2 * (2 * np.pi * freq) ** 2 * step + (1j / 6) * self.beta3 * (
                (2 * np.pi * freq) ** 3 * step) - self.alphalin / 2 * step

        dz_Eff = (1 - np.exp(-self.alphalin * step)) / self.alphalin
        step_number = np.ceil(self.length / step)

        for number in range(int(step_number)):
            print(number)
            if number == step_number - 1:
                # dz = step
                dz = self.length - (step_number - 1) * step
                dz_Eff = (1 - np.exp(-self.alphalin * dz)) / self.alphalin
                Disper = (1j / 2) * self.beta2 * (2 * np.pi * freq) ** 2 * dz + (1j / 6) * self.beta3 * (
                        (2 * np.pi * freq) ** 3 * dz) - self.alphalin / 2 * step
            signal_x, signal_y = self.linear(signal_x, signal_y, Disper)
            energy = signal_x * af.conjg(signal_x) + signal_y * af.conjg(signal_y)
            signal_x, signal_y = self.nonlinear(energy, signal_x, signal_y, dz_Eff)
            signal_x, signal_y = self.linear(signal_x, signal_y, Disper)

        signal_x_array = np.array(signal_x.to_list())
        signal_y_array = np.array(signal_y.to_list())

        signal_x_array = signal_x_array[:, 0] + 1j * signal_x_array[:, 1]

        signal_y_array = signal_y_array[:, 0] + 1j * signal_y_array[:, 1]

        signal.data_sample[0, :] = signal_x_array
        signal.data_sample[1, :] = signal_y_array

    def nonlinear(self, energy, signal_x, signal_y, dz_Eff):
        signal_x = signal_x * af.exp(energy * (8 / 9) * dz_Eff * (1j) * self.gamma)
        signal_y = signal_y * af.exp(energy * (8 / 9) * dz_Eff * (1j) * self.gamma)
        return signal_x, signal_y

    def linear(self, signal_x, signal_y, Disper_cd):
        signal_x_fft = af.fft(signal_x) * af.exp(Disper_cd / 2)
        signal_x = af.ifft(signal_x_fft)

        signal_y_fft = af.fft(signal_y) * af.exp(Disper_cd / 2)
        signal_y = af.ifft(signal_y_fft)
        return signal_x, signal_y

    def __call__(self, sig, step, mode):
        self.prop(sig, step, mode)


class Fiber(object):
    '''
        Fiber contains many span and edfa, traditionally, a edfa is placed after a span

        :param spans list of Span object
        :param edfas list of EDFA object

        the length of edfas and the length of spans should be the same
    '''

    def __init__(self, spans: List[Span], edfas: List[EDFA]):
        self.spans = spans

        self.edfas = edfas

    def prop(self, signal):
        for span, edfa in zip(self.spans, self.edfas):
            span.prop(signal, step=20e-3)
            edfa.prop(signal)


if __name__ == '__main__':
    from tool.plot_const import plot_const

    sig = Signal(35, 4, 1550, 2 ** 16, mf='dp-qpsk')
    sig.generate_data()

    waveshaper = WaveShape(6, 4)
    waveshaper.prop(sig)

    sig.set_signal_power(-2, 'dbm')
    print(sig.signal_power('dbm'))
    span = Span(0.2, -2.1668e-23, 0, 1.3, 80)
    import time

    now = time.time()
    span.prop(sig, 200 * 1e-3)
    print(time.time() - now)
    print(sig.signal_power('dbm'))
    # import matplotlib.pyplot as plt
    # plt.psd(sig.data_sample[0,:],len(sig.data_sample[0,:]),Fs = sig.sps*sig.symbol_rate_in_hz)
    # # plt.show()
    plot_const(sig)
