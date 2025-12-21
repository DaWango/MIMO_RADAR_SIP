from abc import abstractmethod
import logging
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import get_window
from scipy.constants import c

from utility import PlotConfig, PlotType, ScaleDB, measure_time

class Pipeline:
    """
    Processing pipeline for sequential execution of radar signal processing modules.

    The `Pipeline` manages an ordered list of `PipelineModule` instances. Each module
    receives the current data array and a shared metadata dictionary. Modules may
    return either:
    - a new data array, or
    - a tuple (data, metadata_update), where `metadata_update` is merged into the
      global metadata dictionary.

    Parameters
    ----------
    radar_config : object
        Radar configuration object passed to each module upon construction.

    Notes
    -----
    The pipeline executes modules in the order they were added. Each module must
    implement a `run(data, metadata)` method. The metadata dictionary persists
    across all modules and may be extended by any module.

    The initial metadata contains:
        {"orginal_data": data}

    Returns
    -------
    tuple
        (final_data, metadata) where:
        - final_data : ndarray
            Output of the last module in the pipeline.
        - metadata : dict
            Aggregated metadata collected throughout the pipeline.
    """

    def __init__(self):
        self.modules = []

    def add_module(self, module_cls, **kwargs):
        module = module_cls(**kwargs)
        self.modules.append(module)
    
    def run(self, data):
        metadata = {"orginal_data": data}
        for m in self.modules:
            out = m.run(data, metadata)
            if isinstance(out, tuple) and len(out) == 2:
                data, meta_update = out
                if meta_update:
                    metadata.update(meta_update)
            else:
                data = out
        return data, metadata


class PipelineModule:
    """
    Abstract base class for all radar signal‑processing pipeline modules.

    Each module receives the current data array and an optional metadata dictionary.
    Subclasses must implement the `run()` method and return either:
    - a processed data array, or
    - a tuple (data, metadata_update), where `metadata_update` is a dictionary
      merged into the pipeline's global metadata.

    Parameters
    ----------
    radar_config : object
        Radar configuration object provided to all modules for access to system
        parameters, calibration settings, and processing configuration.

    Notes
    -----
    This class defines the interface for all pipeline modules. Concrete subclasses
    must override `run(data, metadata)` and must not modify the input metadata
    in-place; instead, they should return a metadata update dictionary when needed.
    """
    radar_config = None
    logger = logging.getLogger(__name__)

    @abstractmethod
    def run(self, data, metadata=None):
        """
        Process data and optionally metadata.

        Should return either:
        - data (ndarray)  OR
        - (data, metadata_dict)
        """

    @classmethod
    def configurePipeline(cls,radar_config):
        cls.radar_config = radar_config


class RemoveDC_Offset(PipelineModule):
    """
    Pipeline module to remove DC offset from complex IQ ADC data.

    Parameters
    ----------
    radar_config : object
        Radar configuration object (passed to base PipelineModule).
    mode : str, optional
        DC removal mode:
        - 'per_chirp_rx' : remove mean over samples for each (chirp, rx) (default)
        - 'per_rx'       : remove mean over all chirps and samples per rx
        - 'global'       : remove global mean over entire data cube

    Notes
    -----
    Expects `data` shaped as (num_chirps, num_samples, num_rx_antenna)
    and complex dtype. Returns a new array with the same shape and dtype.
    """
    def __init__(self, mode='per_rx'):
        super().__init__()
        self.mode = mode  # 'per_chirp_rx', 'per_rx', or 'global'

    @measure_time
    def run(self, data, metadata=None):
        
        """
        Execute DC offset removal.

        Parameters
        ----------
        data : ndarray
            Complex-valued data cube with shape (num_chirps, num_samples, num_rx_antenna).

        Returns
        -------
        ndarray
            DC-corrected data cube (same shape and dtype as input).
        """
        
        self.logger.info(f"Starting with 'Remove DC Module")
        if not np.iscomplexobj(data):
            data = data.astype(np.complex128)

        if self.mode == 'per_chirp_rx':
            dc = np.mean(data, axis=1, keepdims=True)
            return data - dc

        if self.mode == 'per_rx':
            dc = np.mean(data, axis=(0, 1), keepdims=True)
            return data - dc

        if self.mode == 'global':

            dc = np.mean(data)
            return data - dc

        raise ValueError("mode must be 'per_chirp_rx', 'per_rx' or 'global'.")

class Windowing(PipelineModule):
    """
    Apply a window function to a specified axis of a complex IQ data cube.

    Parameters
    ----------
    radar_config : object
        Radar configuration object (passed to base PipelineModule).
    window : str or array_like, optional
        Window name accepted by scipy.signal.get_window (e.g., 'hann', 'hamming')
        or an array of length matching the axis to window. Default 'hann'.
    axis : int, optional
        Axis to apply the window to. Default 1 (samples per chirp).
    inplace : bool, optional
        If True, modify and return the input array (may save memory).
        If False, operate on a copy. Default False.
    """

    def __init__(self, window='hann', axis=1, inplace=False):
        super().__init__()
        self.window = window
        self.axis = int(axis)
        self.inplace = bool(inplace)
   
    @measure_time
    def run(self, data, metadata=None):
        """
        Execute windowing.

        Parameters
        ----------
        data : ndarray
            Complex-valued data cube, expected shape (num_chirps, num_samples, num_rx).
        Returns
        -------
        ndarray
            Windowed data cube (same shape and dtype as input).
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy ndarray")


        if not np.iscomplexobj(data):
            data = data.astype(np.complex128, copy=False)


        axis_len = data.shape[self.axis]


        if isinstance(self.window, (str, tuple)):
            win = get_window(self.window, axis_len).astype(np.float64)
        else:
            win = np.asarray(self.window, dtype=np.float64)
            if win.shape[0] != axis_len:
                raise ValueError("Provided window length does not match data axis length")


        shape = [1] * data.ndim
        shape[self.axis] = axis_len
        win = win.reshape(shape)

        if self.inplace:
            data *= win
            return data
        else:
            return data * win


class TDMRemapper(PipelineModule):
    """
    Remap chirp-ordered data_cube into virtual-array layout for TDM-MIMO.

    Parameters
    ----------
    radar_config : object
        Radar configuration passed to base PipelineModule. Must provide:
        - num_tx_antenna (int)
        - num_rx_antenna (int)

    Notes
    -----
    Input data shape: (num_chirps, num_samples, num_rx_antenna)

    Output data shape: (chirps_per_tx, num_samples, rx_vitrual)
        chirps_per_tx = K // num_tx
        rx_vitrual = num_tx * M_rx
    """

    def __init__(self):
        super().__init__()
        self.num_tx = int(self.radar_config.num_tx_antenna)
        self.num_rx = int(self.radar_config.num_rx_antenna)

    @measure_time
    def run(self, data, metadata=None):
        """
        Remap data_cube into virtual array layout.

        Parameters
        ----------
        data : ndarray
            Complex data cube with shape (num_chirps, num_samples, num_rx_antenna).
        metadata : dict, optional
            Existing metadata dictionary (ignored here).

        Returns
        -------
        remapped : ndarray
            Remapped cube with shape (chirps_per_tx, num_samples, rx_vitrual).
        meta_update : dict
            Metadata with tx_index_per_chirp, slot_index_per_chirp, x_vitrual_rx.
        """
        if data.ndim != 3:
            raise ValueError("data must have shape (num_chirps, num_samples, num_rx_antenna)")

        num_chirps, num_samples, rx_vitrual = data.shape
        if rx_vitrual != self.num_rx:
            raise ValueError(
                f"data third dim ({rx_vitrual}) != radar_config.num_rx_antenna ({self.num_rx})"
            )

        if num_chirps % self.num_tx != 0:
            raise ValueError("Total chirps num_chirps must be divisible by num_tx")

        chirps_per_tx = num_chirps // self.num_tx
        rx_vitrual = self.num_tx * self.num_rx

        remapped = np.zeros((chirps_per_tx, num_samples, rx_vitrual), dtype=data.dtype)
        tx_index_per_chirp = np.empty(num_chirps, dtype=int)
        slot_index_per_chirp = np.empty(num_chirps, dtype=int)

        for chirps in range(num_chirps):
            tx = chirps % self.num_tx
            slot = chirps // self.num_tx
            tx_index_per_chirp[chirps] = tx
            slot_index_per_chirp[chirps] = slot
            start = tx * self.num_rx
            remapped[slot, :, start:start + self.num_rx] = data[chirps, :, :]

        meta_update = {
            "data_tdm_remapped": remapped,
            "tx_index_per_chirp": tx_index_per_chirp,
            "slot_index_per_chirp": slot_index_per_chirp,
            "x_vitrual_rx": rx_vitrual,
            "num_slots": chirps_per_tx
        }
        return remapped, meta_update

class RangeFFT(PipelineModule):
    """
    Range FFT module for Pipeline.

    Parameters
    ----------
    radar_config : object
        passed to base class (not required for basic FFT).
    axis : int
        axis index of samples per chirp (default 1).
    keep_single_sided : bool
        if True use rfft (default True).
    nfft_range : int or 'samples' or None
        if 'samples' (default) use axis length; if int use that nfft; if None use axis length.
    """

    def __init__(self, axis=1, keep_single_sided=True, nfft_range='samples'):
        super().__init__()
        self.axis = axis
        self.keep_single_sided = keep_single_sided
        self.nfft_range = nfft_range

    @measure_time
    def run(self, data, metadata=None):
        data = np.asarray(data)
        axis_len = data.shape[self.axis]

        if self.nfft_range in ('samples', None):
            nfft = axis_len
        else:

            nfft = int(self.nfft_range)


        fft = np.fft.fft(data, n=nfft, axis=self.axis)

        if self.keep_single_sided:
            
            fft = fft.take(indices=range((nfft // 2)), axis=self.axis)

        old_plot_data = metadata.get("plot_data",[])
        meta_update = {
            "nfft_range": nfft,
            "num_range_bins": fft.shape[self.axis],
            "range_fft_single_sided": self.keep_single_sided,
            "plot_data": old_plot_data + [PlotConfig(plot_type=PlotType.RANGE)],
            "range_fft": fft
        }

        return data, meta_update


class DopplerFFT(PipelineModule):
    def __init__(self):
        super().__init__()

    @measure_time
    def run(self,data,metadata=None): 
        data = np.asarray(data) 
       

        fftout = np.fft.fft2(data,axes=(1,0))

        range_doppler_fft =np.fft.fftshift(fftout,axes=0)


        old_plot_data = metadata.get("plot_data",[])
        meta_update = {
            "plot_data": old_plot_data + [PlotConfig(plot_type=PlotType.RANGE_DOPPLER)],
            "range_doppler_fft" : range_doppler_fft
        }

        return data, meta_update

class PlotModule(PipelineModule):
    """
    Pipeline module for visualizing radar data in 2D or 3D.

    This module generates range–Doppler plots from a processed data cube.
    Depending on configuration, it produces either a 2D magnitude heatmap
    or a 3D surface plot. The module does not modify the data and returns
    it unchanged, serving purely as a visualization stage in the pipeline.

    Parameters
    ----------
    radar_config : object
        Radar configuration object providing system parameters such as
        chirp bandwidth.
    plot_type : str, optional
        Type of plot to generate. Currently supported:
        - 'range' : range–Doppler visualization (default)
    plot3d : bool, optional
        If True, generate a 3D surface plot instead of a 2D heatmap.

    Notes
    -----
    Expects `plot_data` with shape:
        (num_doppler_bins, num_range_bins, num_rx)
    The module currently visualizes only receiver channel index 1.

    Range resolution is computed as:
        delta_r = c / (2 * chirp_bandwidth_hz)

    The module displays the plot immediately using matplotlib and does not
    return any metadata updates.

    Returns
    -------
    ndarray
        The input `plot_data` unchanged.
    """
    def __init__(self):
        super().__init__()

    @measure_time
    def run(self,data,metadata=None):
        
        delta_r = c / (self.radar_config.chirp_bandwidth_hz *2)
        
        for plot in metadata["plot_data"]:
            
        

            match plot.plot_type:
                case PlotType.RANGE: 
                    plot_data = metadata.get("range_fft",data)
                    num_doppler_bins, num_range_bins, num_rx = plot_data.shape
                    x_axis = np.arange(num_doppler_bins)
                    label_x = "Doppler-Bins"
                    label_y = "Range [m]"
                case PlotType.RANGE_DOPPLER:
                    plot_data = metadata.get("range_doppler_fft",data)
                    num_doppler_bins, num_range_bins, num_rx = plot_data.shape
                    detla_v = self.radar_config.lambda_ / ((self.radar_config.T_k *self.radar_config.num_tx_antenna)  *4 * num_doppler_bins )
                    v_max = (num_doppler_bins/2) * detla_v
                    x_axis = np.linspace(-v_max,v_max- detla_v,num_doppler_bins)
                    #k = np.arange(num_doppler_bins) - num_doppler_bins//2
                    #f_d= k *1/(self.radar_config.T_k* num_doppler_bins)
                    #x_axis = f_d * (self.radar_config.lambda_/2)
                    label_x = "Velocity [m/s]"
                    label_y = "Range [m]"
                
                case _:
                    raise ValueError(f"unknown plot_type selected")
            
            y_axis = np.arange(num_range_bins) * delta_r
            
            match plot.scale_db:
                case ScaleDB.AMPLITUDE:
                    db_factor = 20
                    label_z = "Amplidute [db]"
                case ScaleDB.POWER:
                    db_factor = 10
                    label_z = "Power [db]"
                case _:
                    raise ValueError(f"unknown ScaleDB selected")

            scaled_data = db_factor * np.log10(np.abs(plot_data[:,:,plot.for_rx]).T +1e-12)

            if not plot.plot3d:

                plt.imshow(scaled_data,
                           extent=[
                                x_axis[0],
                                x_axis[-1],
                                y_axis[0],
                                y_axis[-1]
                            ],
                           origin='lower', aspect='auto', cmap='viridis')
                plt.colorbar(label=label_z)
                plt.xlabel(label_x)
                plt.ylabel(label_y)
                plt.show()
            
            else:
                X, Y = np.meshgrid(x_axis, y_axis, indexing='ij')
                fig = plt.figure(figsize=(12, 7))
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(X, Y, scaled_data,
                                       cmap='viridis')
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=label_z)
                ax.set_xlabel(label_x)
                ax.set_ylabel(label_y)
                ax.set_zlabel(label_z)
                plt.show()