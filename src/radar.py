import os
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import c, pi
from dataclasses import dataclass 
import logging

from sip import *


class Radar():
    def __init__(self):
        config_path=os.path.join(os.path.dirname(__file__),"..","config","radar_config.json")
        self.param = RadarConfig.from_json(config_path)
        self.frontend = RadarFrontend(self.param)
        self.backend = RadarBackend(self.param)

    def power_on(self,targets):
        data = self.frontend.start_recording(targets)
        self.backend.start_sip(data=data)

class RadarFrontend(Radar):

    def __init__(self, param):
        self.log = logging.getLogger(__name__)
        self.param = param
        self.radar_pos = np.zeros(3)
        self.A_tx = 1 # TODO calculate A_tx dep on power
        self.A_rx = 0.8 # TODO calculate A_rx with radar eq
        self.f_c = self.param.carrier_frequency_hz + (self.param.chirp_bandwidth_hz*0.5)
        self.T_k = self.param.num_samples_per_chirp / self.param.sampling_rate_hz
        self.m = self.param.chirp_bandwidth_hz / self.T_k
        self.dt = 1 / self.param.sampling_rate_hz
        self.lambda_ = c/self.f_c
        self.tx_antennas = list[RadarAntenna]
        self.rx_antennas = list[RadarAntenna]
        self._set_antennas(2,2,4,4)

    def __repr__(self):
        return (f"{self.param!r}")
    
    def _generate_if_signal_at_t(self,t,t_index,k,targets,rx_antenna):
        phi_mulitplier = (2*pi / c)
        cur_tx_antenna =(k % self.param.num_tx_antenna) 
        u_t = 0
        for tgt in targets:
            dist_comb = tgt.distance_to_tx[t_index,cur_tx_antenna] + tgt.distance_to_rx[t_index,rx_antenna]
            phi_r0 = self.f_c*dist_comb
            phi_dop = 2* self.f_c *tgt.rel_velocity_m_s[t_index]*self.T_k*k
            phi_dist = self.m *dist_comb*t
            phi_total = phi_mulitplier * (phi_r0+phi_dop+phi_dist)
            u_t += 0.5 * self.A_tx * self.A_rx * np.exp(1j * phi_total)
        return self.add_awgn(u_t,self.param.awgn_snr_db)

    def start_recording(self,with_targets):
        
        data_cube = np.zeros((self.param.num_chirps, self.param.num_samples_per_chirp, self.param.num_rx_antenna), dtype=complex)
        t_index  = 0
        self.log.info(f"Need to calculate {len(with_targets)*self.param.num_rx_antenna*self.param.num_chirps*self.param.num_samples_per_chirp} updates")
        for chirp in range(self.param.num_chirps):
            for cur_sample in range(self.param.num_samples_per_chirp):
                t = cur_sample * self.dt
                for rx_antenna in range(self.param.num_rx_antenna):
                    data_cube[chirp,cur_sample,rx_antenna] = self._generate_if_signal_at_t(t,t_index,chirp,with_targets,rx_antenna)
                t_index += 1
        self.log.info(f"data is ready")

        
        return data_cube
        
        

    def add_awgn(self,iq_data, snr_db):
        """
        Additive White Gaussian Noise (AWGN) to IQ data.
        snr_db: desired Signal-to-Noise Ratio in dB
        """
        # Calculate average signal power
        sig_power = np.mean(np.abs(iq_data)**2)

        # Convert SNR from dB to linear scale
        snr_linear = 10**(snr_db/10)

        # Calculate noise power based on desired SNR
        noise_power = sig_power / snr_linear

        # Generate complex Gaussian noise (real + imaginary parts)
        noise = np.sqrt(noise_power/2) * (
            np.random.randn(*iq_data.shape) + 1j*np.random.randn(*iq_data.shape)
        )

        # Return noisy IQ data
        return iq_data + noise


    def plot_range_time(self,data_cube, rx_index=0, radar_param=None):
        rx_data = data_cube[:, :, rx_index]  # [chirp, sample]
        range_fft = np.fft.fft(rx_data, axis=1)
        amplitude = np.abs(range_fft[:, :range_fft.shape[1]//2])

        if radar_param is not None:
            fs = self.param.sampling_rate_hz
            B = self.param.chirp_bandwidth_hz
            N = self.param.num_samples_per_chirp
            c = 3e8
            freqs = np.fft.fftfreq(N, d=1/fs)[:N//2]
            ranges = freqs * c / (2 * B)
        else:
            ranges = np.arange(amplitude.shape[1])

        plt.figure(figsize=(8, 6))
        plt.imshow(amplitude.T,
                   aspect='auto',
                   origin='lower',
                   extent=[0, amplitude.shape[0], ranges[0], ranges[-1]],
                   cmap='viridis')
        plt.colorbar(label="Amplitude")
        plt.xlabel("Chirp Index")
        plt.ylabel("Range [m]" if radar_param else "Range Bin")
        plt.title("Range-Time-Amplitude Plot")
        plt.show()




    def _set_antennas(self, n_tx_x, n_tx_y, n_rx_x, n_rx_y):

        d_rx = self.lambda_ / 2
        d_tx_x = n_rx_x * d_rx
        d_tx_y = n_rx_y * d_rx
        
        rx_coords = np.array([
            [(i+1)*d_rx + d_tx_x, j*d_rx, 0]
            for i in range(n_rx_x) for j in range(n_rx_y)
        ], dtype=float)

        self.rx_antennas = [
            RadarAntenna(idx+1, rx_coords[idx])
            for idx in range(rx_coords.shape[0])
        ]

        tx_coords = np.array([
            [i*d_tx_x, j*d_tx_y, 0]
            for i in range(n_tx_x) for j in range(n_tx_y)
        ], dtype=float)

        self.tx_antennas = [
            RadarAntenna(idx+1, tx_coords[idx])
            for idx in range(tx_coords.shape[0])
        ]
        

    def plot_antennas(self):
        tx_coords = np.array([ant.rel_pos_to_frontend for ant in self.tx_antennas])
        rx_coords = np.array([ant.rel_pos_to_frontend for ant in self.rx_antennas])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(tx_coords[:,0], tx_coords[:,1], tx_coords[:,2], c='r', marker='o', label='TX')
        ax.scatter(rx_coords[:,0], rx_coords[:,1], rx_coords[:,2], c='b', marker='^', label='RX')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

@dataclass
class RadarConfig():
    awgn_snr_db : int = 0
    carrier_frequency_hz : int = 0
    chirp_bandwidth_hz : int = 0
    sampling_rate_hz : int = 0
    num_samples_per_chirp : int = 0
    num_chirps : int = 0
    num_tx_antenna : int = 0
    num_rx_antenna : int = 0

    @classmethod
    def from_json(cls, json_path):
        with open(json_path,'r') as jf:
            config_data = json.load(jf)
        radar_config = config_data["radar_frontend"]
        sim_config = config_data["sim_config"]
        return cls(
            awgn_snr_db = sim_config.get("awgn_snr_db"),
            carrier_frequency_hz = radar_config.get("carrier_frequency_hz",0),
            chirp_bandwidth_hz = radar_config.get("chirp_bandwidth_hz",0),
            sampling_rate_hz = radar_config.get("sampling_rate_hz",0),
            num_samples_per_chirp = radar_config.get("num_samples_per_chirp",0),
            num_chirps = radar_config.get("num_chirps",0),
            num_tx_antenna = radar_config.get("num_tx_antenna",0),
            num_rx_antenna = radar_config.get("num_rx_antenna",0)
        )
        
class RadarAntenna():
    def __init__(self,number,rel_pos_to_frontend):
        self.number = number
        self.rel_pos_to_frontend = rel_pos_to_frontend

class RadarBackend():
    def __init__(self, radar_config):
        self.pipeline = Pipeline(radar_config=radar_config)
        self.pipeline.add_module(RemoveDC_Offset)
        self.pipeline.add_module(Windowing)
        self.pipeline.add_module(TDMRemapper)
        self.pipeline.add_module(RangeFFT, axis=1, keep_single_sided=True, nfft_range='samples')
        self.pipeline.add_module(PlotModule,plot_type='range')



        
    def start_sip(self, data):
        self.pipeline.run(data)
