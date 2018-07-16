function Output_Samples =cd(Input_Samples, DTime, Fiber_Length, Beta2,varargin)


    p = inputParser;
    p.addParameter('CUDA', 1);
    p.parse(varargin{:});
    Number_of_Samples = length(Input_Samples);
    Opt = p.Results;
%     MaxFreq = 0.5/DTime;
%     DFreq = 2*MaxFreq/(Number_of_Samples-1);
%
%     VFreq = (-1:Number_of_Samples-2)*DFreq;
%     VFreq = VFreq-0.5*max(VFreq);
%     VOmeg = 2*pi*VFreq;
%

    if mod(Number_of_Samples , 2) == 0

        VFreq = [(0:Number_of_Samples/2-1),(-Number_of_Samples/2: -1)] / (Number_of_Samples*DTime) ;
        %     VFreq(VFreq<0)  = VFreq(VFreq<0)+1/DTime;
        VOmeg = 2*pi*VFreq;
    else
        VFreq = [(0:(Number_of_Samples-1)/2),(-(Number_of_Samples-1)/2: -1)] / (Number_of_Samples*DTime) ;
        %     VFreq(VFreq<0)  = VFreq(VFreq<0)+1/DTime;
        VOmeg = 2*pi*VFreq;
    end

    %Dispersion Parameter

    if Opt.CUDA
       VOmeg = gpuArray(VOmeg);

       Input_Samples = gpuArray(Input_Samples);
    end
    Disper = -(1j/2)*Beta2*VOmeg.^2*Fiber_Length;
    %dispersion
%     Freq_Samples = fftshift(fft(Input_Samples.').');
%     Output_Freq_Samples = Freq_Samples.*exp(Disper);
%     Output_Samples = ifft(ifftshift(Output_Freq_Samples).').';
    Freq_Samples = fft(Input_Samples.').';
    Output_Freq_Samples = Freq_Samples.*exp(Disper);
    Output_Samples = ifft(Output_Freq_Samples.').';

    Output_Samples = gather(Output_Samples);
end

