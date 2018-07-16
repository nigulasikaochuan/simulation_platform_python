import sys
print(sys.path)
from numpy import random, roll
from scipy.signal import upfirdn

from common_package import *
import matlab.engine

ENG = matlab.engine.connect_matlab()



class Signal:
    '''
        This Class Complete Signal Class. the object of this class represent 
        optical signal during transimission,and the property of the object is:
            
            symbol_rate: represent symbol rate of this signal [GBAUD]
            sps        : represent the number per symbol 
            center_lambda : represent the lambda of optical carrier
            data_len : represent the message length
            msg:   repersent 10-jinzhi raw message
            symbol : represent the modulated symbol
            data_sample: will be set automatically after wave shaping
        
        methods:
            def symbol_rate_in_hz(self): return symbol rate with unit BAUD
            
            def center_lambda_in_m(self): return optical carrier's lambda in m
            
            def signal_power(self, unit='w'): return signal power, unit can be
            w or dbm
            
            def generate_data(self):change message in to qpsk or other modulation's format
            
            
            def set_signal_power(self, p, unit='w'): change signal's power
    '''

    def __init__(self, symbol_rate, sps, center_lambda, data_len, mf):
        self.symbol_rate = symbol_rate
        self.sps = sps
        self.center_lambda = center_lambda
        self.data_len = data_len
        self.mf = mf

        self.msg = None
        self.symbol = None
        self.data_sample = None

    @property
    def symbol_rate_in_hz(self):
        return self.symbol_rate * 1e9

    @property
    def center_lambda_in_m(self):
        return self.center_lambda * 1e-9
    
    def signal_power(self,unit='w'):
        '''
            caculate the signal power of each polarization,self.data_sample each pol is a row
        '''
        x_pol = self.data_sample[0,:]
        y_pol = self.data_sample[1,:]
        x_power = sum(x_pol*np.conj(x_pol))/len(x_pol)
        y_power = sum(y_pol*np.conj(y_pol))/len(y_pol)
        power = x_power+y_power
        if unit =='w':
            return power
        elif unit=='dbm':
            return 10*np.log10(power*1000)
        else:
            print('unit error')
        
#    
#    def signal_power(self, unit='w'):
#        power = np.mean(abs(self.data_sample[0, :]) ** 2) + np.mean(abs(self.data_sample[1, :]) ** 2)
#        if unit == 'w':
#            return power
#        if unit == 'dbm':
#            return 10 * np.log10(power / 1e-3)

    def generate_data(self):

        if self.mf == 'dp-qpsk':
            self.const = np.exp([1j * np.pi / 4, 1j * np.pi / 4 * (-1), 1j * np.pi / 4 * 3, 1j * np.pi / 4 * (-3)])
            msges = random.randint(0, 4, (2, self.data_len))
            self.symbol = np.zeros_like(msges, dtype=np.complex)

            for msg in range(4):
                self.symbol[msges == msg] = self.const[msg]

            self.msg = msges

    def set_signal_power(self, p, unit='w'):
        if unit == 'w':
            self.data_sample = self.data_sample * np.sqrt(p / sum(np.mean(abs(self.data_sample) ** 2, 1)))
        if unit == 'dbm':
            self.data_sample = self.data_sample * np.sqrt(
                ((10 ** (p / 10)) / 1000) / sum(np.mean(abs(self.data_sample) ** 2, 1)))


class WaveShape:
    '''
        wave shaping,rc or rrc,need matlab supported
    '''

    def __init__(self, span, sps, roll_off=0.2,shape = 'normal'):
        self.span = span
        self.sps = sps
        self.roll_off = roll_off
       
        self.shape = shape
        self.h = self.design()
    def prop(self, signal: Signal):
        from scipy.signal import upfirdn

        signal.data_sample = upfirdn(self.h, signal.symbol, signal.sps, 1)
        signal.data_sample = roll(signal.data_sample, int(-self.span / 2 * self.sps), axis=-1)

        tempx = signal.data_sample[0, 0:signal.sps * signal.data_len]
        tempy = signal.data_sample[1, 0:signal.sps * signal.data_len]

        signal.data_sample = np.array([tempx, tempy])

    def design(self):
        h = ENG.rcosdesign(self.roll_off, float(self.span),
                           float(self.sps), self.shape,nargout=1)
        h = np.array(h)
        h = h/np.max(h[0])
        return h[0]


class Decoder(object):

    def prop(self, signal: Signal):
        recv_symbol_x = []
        recv_symbol_y = []
        msg_x = []
        msg_y = []
        for recv_x in signal.data_sample[0, :]:
            edu_distance = recv_x - signal.const
            edu_distance = edu_distance * np.conj(edu_distance)
            index = np.argmin(edu_distance)
            recv_symbol_x.append(signal.const[index])
            msg_x.append(Decoder.demap(signal.mf, index))
        for recv_y in signal.data_sample[1, :]:
            edu_distance = recv_y - signal.const
            edu_distance = edu_distance * np.conj(edu_distance)
            index = np.argmin(edu_distance)
            recv_symbol_y.append(signal.const[index])

            msg_y.append(Decoder.demap(signal.mf, index))
        return np.array(msg_x), np.array(msg_y), np.array(recv_symbol_x), np.array(recv_symbol_y)

    @staticmethod
    def demap(mf, index):

        if mf == 'dp-qpsk':
            msg = [0, 1, 2, 3]
            return msg[index]


if __name__ == '__main__':
    sig = Signal(35, 4, 1550, 2 ** 4, 'dp-qpsk')
    sig.generate_data()

    spans = 6
    spss = 4

    shaper = WaveShape(spans, spss)
    shaper.prop(sig)

    tempx = sig.data_sample[0,0:sig.sps*sig.data_len]
    tempy= sig.data_sample[1, 0:sig.sps * sig.data_len]

    sig.data_sample = np.array([tempx,tempy])
    sig.data_sample = upfirdn([1], sig.data_sample, 1, spss)
    decod = Decoder()
    print(decod.prop(sig)[0], decod.prop(sig)[1])
