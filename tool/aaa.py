def _split_fourier_cuda(self, signal: Signal, step):
    af.set_backend('cuda')

    freq = cp.fft.fftfreq(len(signal.data_sample[0, :]), (signal.sps * signal.symbol_rate_in_hz) ** (-1))

    signal_x = cp.asarray(signal.data_sample[0, :])
    signal_y = cp.asarray(signal.data_sample[1, :])

    Disper = -(1j / 2) * self.beta2 * (2 * cp.pi * freq) ** 2 * step + (1j / 6) * self.beta3 * (
            (2 * np.pi * freq) ** 3 * step)

    dz_Eff = (1 - cp.exp(-self.alphalin * step)) / self.alphalin
    step_number = cp.ceil(self.length / step)

    for number in range(int(step_number)):
        print(number)
        if number == step_number - 1:
            # dz = step
            dz = self.length - (step_number - 1) * step
            dz_Eff = (1 - np.exp(-self.alphalin * step)) / self.alphalin
            Disper = -(1j / 2) * self.beta2 * (2 * np.pi * freq) ** 2 * dz + (1j / 6) * self.beta3 * (
                    (2 * np.pi * freq) ** 3 * dz)

        signal_x = signal_x * cp.exp(-self.alphalin / 2 * step)
        signal_y = signal_y * cp.exp(-self.alphalin / 2 * step)

        signal_x, signal_y = self.linear(signal_x, signal_y, Disper)
        energy = signal_x * cp.conj(signal_x) + signal_y * cp.conj(signal_y)

        self.nonlinear(energy, signal_x, signal_y, dz_Eff)

        self.linear(signal_x, signal_y, Disper)

        signal_x = signal_x * cp.exp(-self.alphalin / 2 * step)
        signal_y = signal_y * cp.exp(-self.alphalin / 2 * step)

    signal.data_sample[0, :] = cp.array(signal_x.T.to_list())
    signal.data_sample[1, :] = cp.array(signal_y.T.to_list())


def nonlinear(self, energy, signal_x, signal_y, dz_Eff):
    signal_x *= signal_x * cp.exp(energy * (8 / 9) * dz_Eff * (1j) * self.gamma * (-1))
    signal_y *= signal_y * cp.exp(energy * (8 / 9) * dz_Eff * (1j) * self.gamma * (-1))


def linear(self, signal_x, signal_y, Disper_cd):
    signal_x_fft = cp.fft.fft(signal_x) * cp.exp(Disper_cd)
    signal_x = cp.fft.ifft(signal_x_fft)

    signal_y_fft = cp.fft.fft(signal_y) * cp.exp(Disper_cd)
    signal_y = cp.fft.ifft(signal_y_fft)

    return signal_x, signal_y