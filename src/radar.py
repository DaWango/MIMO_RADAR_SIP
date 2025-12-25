import os
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import c, pi
from dataclasses import dataclass 
import logging

from sip import *
from utility import is_perfect_square


class Radar():
    def __init__(self):
        config_path=os.path.join(os.path.dirname(__file__),"..","config","radar_config.json")
        self.radar_config = RadarConfig.from_json(config_path)
        
        self.frontend = RadarFrontend(self.radar_config)
        self.backend = RadarBackend(self.radar_config)

    def power_on(self,targets):
        data = self.frontend.start_recording(targets)
        self.backend.start_sip(data=data)

class RadarFrontend:
    """
    Frontend component for simulating FMCW radar IF signals.

    The `RadarFrontend` generates a full radar data cube by simulating the
    intermediate‑frequency (IF) signal received at each RX antenna for each
    chirp and sample. The model includes FMCW phase terms, Doppler shift,
    target motion, antenna geometry, and additive white Gaussian noise (AWGN).

    Parameters
    ----------
    radar_config : object
        Radar configuration object providing system parameters such as
        carrier frequency, chirp bandwidth, sampling rate, number of chirps,
        and antenna counts.

    Attributes
    ----------
    radar_pos : ndarray, shape (3,)
        Cartesian position of the radar in meters.
    A_tx : float
        Transmit amplitude scaling factor (placeholder, may be computed from
        transmit power in future versions).
    A_rx : float
        Receive amplitude scaling factor (placeholder, may be computed from
        radar equation in future versions).
    f_c : float
        Effective carrier frequency (center of chirp bandwidth).
    T_k : float
        Chirp duration in seconds.
    m : float
        FMCW chirp slope (Hz/s).
    dt : float
        Sampling interval in seconds.
    lambda_ : float
        Wavelength corresponding to `f_c`.
    tx_antennas : list of RadarAntenna
        List of transmit antenna objects with 3D coordinates.
    rx_antennas : list of RadarAntenna
        List of receive antenna objects with 3D coordinates.

    Notes
    -----
    The IF signal for each target is modeled as:

        u(t) = A * exp(j * phi_total)

    where the total phase consists of:
        - phi_r0  : phase due to round‑trip distance
        - phi_dop : Doppler phase due to relative velocity
        - phi_dist: FMCW beat frequency term (chirp slope * distance * time)

    The frontend simulates:
        - MIMO TX cycling (k % num_tx_antenna)
        - per‑sample time evolution
        - per‑antenna geometry
        - AWGN based on desired SNR

    The output is a complex data cube with shape:
        (num_chirps, num_samples_per_chirp, num_rx_antenna)

    Methods
    -------
    start_recording(with_targets)
        Generate a full radar data cube for the provided list of targets.
    add_awgn(iq_data, snr_db)
        Add complex AWGN to the simulated IQ sample.
    """


    def __init__(self, radar_config):
        self.log = logging.getLogger(__name__)
        self.radar_config = radar_config
        self.radar_pos = np.zeros(3)
        self.A_tx = .4 # TODO calculate A_tx dep on power
        self.A_rx = 0.2 # TODO calculate A_rx with radar eq
        self.m = self.radar_config.chirp_bandwidth_hz / self.radar_config.T_k
        self.dt = 1 / self.radar_config.sampling_rate_hz
        self.tx_antennas = list[RadarAntenna]
        self.rx_antennas = list[RadarAntenna]

        #TODO: refactor antennen postions 
        is_perfect, tx = is_perfect_square(self.radar_config.num_tx_antenna)

        if not is_perfect:
            raise ValueError(f"atm only perfect sqrt values for num_tx are valid")

        is_perfect, rx = is_perfect_square(self.radar_config.num_rx_antenna)

        if not is_perfect:
            raise ValueError(f"atm only perfect sqrt values for num_rx are valid")
        
        self._set_antennas(tx,tx,rx,rx)

    def __repr__(self):
        return (f"{self.radar_config!r}")
    
    def _generate_if_signal_at_t(self,t,t_index,k,targets,rx_antenna):
        phi_mulitplier = (2*pi / c)
        cur_tx_antenna =(k % self.radar_config.num_tx_antenna) 
        u_t = 0
        for tgt in targets:
            dist_comb = tgt.distance_to_tx[t_index,cur_tx_antenna] + tgt.distance_to_rx[t_index,rx_antenna]
            phi_r0 = self.radar_config.f_c*dist_comb
            phi_dop = 2* self.radar_config.f_c *tgt.rel_velocity_m_s[t_index]*self.radar_config.T_k*k
            phi_dist = self.m *dist_comb*t
            phi_total = phi_mulitplier * (phi_r0+phi_dop+phi_dist)
            u_t += 0.5 * self.A_tx * self.A_rx * np.exp(1j * phi_total)
        return self.add_awgn(u_t,self.radar_config.awgn_snr_db)

    @measure_time
    def start_recording(self,with_targets):
        
        data_cube = np.zeros((self.radar_config.num_chirps, self.radar_config.num_samples_per_chirp, self.radar_config.num_rx_antenna), dtype=complex)
        t_index  = 0
        self.log.info(f"Need to calculate {len(with_targets)*self.radar_config.num_rx_antenna*self.radar_config.num_chirps*self.radar_config.num_samples_per_chirp} updates")
        for chirp in range(self.radar_config.num_chirps):
            for cur_sample in range(self.radar_config.num_samples_per_chirp):
                t = cur_sample * self.dt
                for rx_antenna in range(self.radar_config.num_rx_antenna):
                    data_cube[chirp,cur_sample,rx_antenna] = self._generate_if_signal_at_t(t,t_index,chirp,with_targets,rx_antenna)
                t_index += 1
        self.log.info(f"data is ready")

        
        return data_cube
        
    def add_awgn(self,iq_data, snr_db):
        """
        Additive White Gaussian Noise (AWGN) to IQ data.
        snr_db: desired Signal-to-Noise Ratio in dB
        """
        sig_power = np.mean(np.abs(iq_data)**2)

        snr_linear = 10**(snr_db/10)

        noise_power = sig_power / snr_linear

        noise = np.sqrt(noise_power/2) * (
            np.random.randn(*iq_data.shape) + 1j*np.random.randn(*iq_data.shape)
        )

        return iq_data + noise



    def _set_antennas(self, n_tx_x, n_tx_y, n_rx_x, n_rx_y):

        d_rx = self.radar_config.lambda_ / 2
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


        

@dataclass
class RadarConfig():
    """
    Container for all radar and simulation configuration parameters.

    This dataclass stores the essential system parameters required by the
    radar frontend, backend, and processing pipeline. It can be constructed
    directly or loaded from a JSON configuration file via `from_json()`.

    Parameters
    ----------
    awgn_snr_db : int
        Desired SNR level for AWGN injection in the simulated IQ data.
    carrier_frequency_hz : int
        Radar carrier frequency in Hz.
    chirp_bandwidth_hz : int
        FMCW chirp bandwidth in Hz.
    sampling_rate_hz : int
        ADC sampling rate in Hz.
    num_samples_per_chirp : int
        Number of ADC samples per chirp.
    num_chirps : int
        Number of chirps per frame.
    num_tx_antenna : int
        Number of transmit antennas.
    num_rx_antenna : int
        Number of receive antennas.

    Notes
    -----
    The JSON loader expects a structure with two sections:
        - "radar_frontend" : radar system parameters
        - "sim_config"     : simulation‑specific parameters (e.g., SNR)

    Example JSON structure:
        {
            "radar_frontend": { ... },
            "sim_config": { ... }
        }
    """

    awgn_snr_db : int = 0
    carrier_frequency_hz : int = 0
    chirp_bandwidth_hz : int = 0
    sampling_rate_hz : int = 0
    num_samples_per_chirp : int = 0
    num_chirps : int = 0
    num_tx_antenna : int = 0
    num_rx_antenna : int = 0
    f_c : int = 0
    T_k : int = 0 
    lambda_ : int = 0

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
            num_rx_antenna = radar_config.get("num_rx_antenna",0),
            f_c = radar_config.get("carrier_frequency_hz",0) + (radar_config.get("chirp_bandwidth_hz",1) * 0.5),
            T_k = radar_config.get("num_samples_per_chirp") / radar_config.get("sampling_rate_hz"),
            lambda_ = c/(radar_config.get("carrier_frequency_hz",0) + (radar_config.get("chirp_bandwidth_hz",1) * 0.5))
        )
        
class RadarAntenna():
    """
    Representation of a radar antenna element with a fixed position.

    Parameters
    ----------
    number : int
        Antenna index within the TX or RX array.
    rel_pos_to_frontend : array_like, shape (3,)
        3D position of the antenna relative to the radar frontend origin,
        expressed in meters.

    Notes
    -----
    Antenna positions are typically assigned by the frontend during array
    construction. This class serves as a lightweight container for geometry
    information used in signal simulation and MIMO processing.
    """

    def __init__(self,number,rel_pos_to_frontend):
        self.number = number
        self.rel_pos_to_frontend = rel_pos_to_frontend

class RadarBackend():
    """
    Backend processing chain for radar signal processing.

    The backend constructs a `Pipeline` consisting of sequential processing
    modules such as DC removal, windowing, TDM remapping, FFT processing,
    and visualization. The backend receives raw IQ data from the frontend
    and executes the full processing chain.

    Parameters
    ----------
    radar_config : RadarConfig
        Configuration object passed to all pipeline modules.

    Notes
    -----
    The default pipeline includes:
        - RemoveDC_Offset
        - Windowing
        - TDMRemapper
        - RangeFFT
        - PlotModule

    Additional modules can be added or removed by modifying the pipeline
    construction in `__init__()`.

    Methods
    -------
    start_sip(data)
        Execute the full signal‑processing pipeline on the provided data cube.
    """

    def __init__(self, radar_config):
        self.pipeline = Pipeline()
        PipelineModule.configurePipeline(radar_config)
        self.pipeline.add_module(RemoveDC_Offset)
        self.pipeline.add_module(Windowing)
        self.pipeline.add_module(TDMRemapper)
        self.pipeline.add_module(RangeFFT, axis=1, keep_single_sided=True, nfft_range='samples')
        self.pipeline.add_module(DopplerFFT)
        self.pipeline.add_module(CaCfarModule)
        self.pipeline.add_module(PlotModule)

    def start_sip(self, data):
        self.pipeline.run(data)
