from elements.EDFA import EDFA
from common_package import *
from Signal.Signal import WaveShape
from Signal.Signal import Signal
from elements.Fiber import Span

from tool.plot_const import plot_const


def cd_compensation(signal: Signal, span: Span):
    '''

    :param signal: signal to compensate
    :param span: Span object
    :return:
    '''
    freq = fftfreq(len(signal.data_sample[0, :]), (signal.sps * signal.symbol_rate_in_hz) ** (-1))
    omeg = 2 * np.pi * freq
    Disper = -(1j / 2) * span.beta2 * omeg ** 2 * span.length
    x_fft = fft(signal.data_sample[0, :]) * np.exp(Disper)
    signal.data_sample[0, :] = ifft(x_fft)

    y_fft = fft(signal.data_sample[1, :]) * np.exp(Disper)
    signal.data_sample[1, :] = ifft(y_fft)


if __name__ == '__main__':
    '''
          self.mode = mode
            self.gain = gain
            self.desired_power = desired_power
            self.nf = nf
    '''
    edfa = EDFA(mode='heihei',gain = 16,nf=5)
    symbolrate=32
    sps=4
    data_len=2**16
    sig = Signal(symbol_rate=symbolrate,sps=sps,center_lambda=1550,data_len=data_len,mf='dp-qpsk')
    sig.generate_data()

    span=6
    wave = WaveShape(span,sps)
    wave.prop(sig)
    #设置传播信号的功率
    sig.set_signal_power(2,'dbm')
    ###########################
    span  = Span(0.2, -2.1668e-23,0,1.3,80)
    span.prop(sig,20/1000)
    edfa.prop(sig)
    cd_compensation(sig,span)
    plot_const(sig)

