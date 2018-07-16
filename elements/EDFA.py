from Signal import Signal
from common_package import *
from utility.power import power_meter
from common_package.constant import PLANK
from common_package.constant import CLIGHT
from utility import check_array

class EDFA:
    '''
        ideal Edfa with ase noise,there two mode of EDFA object, one is fixed-power, one is fixed-gain, when the mode is
        fixed power, the desired_power property of EDFA object must be set

        the parameter of EDFA object:
            :param mode : str fixed-power or fixed-gain
            :param gain : in db
            :param desired_power : when mode is fixed-power, the output power of EDFA object
            :param nf: noise figure of this edfa

        the unit of ase power will be W

        when use, just obj(signal)

    '''
    def __init__(self, mode, gain, nf, desired_power=None):

        self.mode = mode
        self.gain = gain
        self.desired_power = desired_power
        self.nf = nf

    def _adjust_gain(self, power_of_signal, desired_power):
        '''
            this function is used to jisuan gain automatically,when the mode of EDFA is fixed-power.
            and will be called in prop,should not be used outside
        :param power_of_signal:
        :param desired_power:
        :return:
        '''
        pass

    def prop(self, signal):

        if self.mode == 'fixed-power':
            power_of_signal = power_meter(signal)
            self._adjust_gain(power_of_signal, self.desired_power)
            signal.data_sample = signal.data_sample * np.sqrt(self.gain_lin)

        else:
            signal.data_sample = signal.data_sample * np.sqrt(self.gain_lin)
        self.add_ase(signal)

    @property
    def gain_lin(self):

        return 10 ** (self.gain / 10)

    @property
    def nf_lin(self):

        return 10 ** (self.gain / 10)

    def add_ase(self, signal:Signal):
        '''

        :param signal:the edfa's input signal， if the signal is wdm, then the lambda will use
        center lambda, correction needed in the future
        :return:
        '''

        if check_array.isscalar(signal.center_lambda_in_m):
            sigma = np.sqrt(self.nf_lin * PLANK * CLIGHT / signal.center_lambda_in_m *
                            (self.gain_lin - 1) * signal.sps * signal.symbol_rate_in_hz)
        else:
            max_lam = np.max(signal.center_lambda_in_m)
            min_lam = np.min(signal.center_lambda_in_m)
            center = 2*max_lam*min_lam/(max_lam+min_lam)
            sigma = np.sqrt(self.nf_lin * PLANK * CLIGHT / center *
                            (self.gain_lin - 1) * signal.sps * signal.symbol_rate_in_hz)

        noise = 0.5*sigma * (np.random.randn(2, signal.sps * signal.data_len) +
                               1j * np.random.randn(2, signal.sps * signal.data_len))

        signal.data_sample = signal.data_sample + noise

    def __call__(self, signal):

        self.prop(signal)