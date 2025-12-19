import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.constants import c, pi

class Pipeline:
    def __init__(self, radar_config):
        self.radar_config = radar_config
        self.modules = []

    def add_module(self, module_cls, **kwargs):
        module = module_cls(self.radar_config, **kwargs)
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
    def __init__(self, radar_config):
        self.radar_config = radar_config

    def run(self, data, metadata=None):
        """
        Process data and optionally metadata.

        Should return either:
        - data (ndarray)  OR
        - (data, metadata_dict)
        """
        raise NotImplementedError



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
    def __init__(self, radar_config, mode='per_rx'):
        super().__init__(radar_config)
        self.mode = mode  # 'per_chirp_rx', 'per_rx', or 'global'

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

    def __init__(self, radar_config, window='hann', axis=1, inplace=False):
        super().__init__(radar_config)
        self.window = window
        self.axis = int(axis)
        self.inplace = bool(inplace)

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

    def __init__(self, radar_config):
        super().__init__(radar_config)
        self.num_tx = int(radar_config.num_tx_antenna)
        self.num_rx = int(radar_config.num_rx_antenna)

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

    def __init__(self, radar_config, axis=1, keep_single_sided=True, nfft_range='samples'):
        super().__init__(radar_config)
        self.axis = axis
        self.keep_single_sided = keep_single_sided
        self.nfft_range = nfft_range

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

        meta_update = {
            "nfft_range": nfft,
            "num_range_bins": fft.shape[self.axis],
            "range_fft_single_sided": self.keep_single_sided,
        }

        return fft, meta_update
    

class PlotModule(PipelineModule):
    def __init__(self, radar_config,plot_type="range",plot3d=False):
        super().__init__(radar_config)
        self.plot_type = plot_type
        self.plot3d = plot3d

    def run(self,plot_data,metadata=None):
        num_doppler_bins, num_range_bins, num_rx = plot_data.shape
        delta_r = c / (self.radar_config.chirp_bandwidth_hz *2)
        
        match self.plot_type:
            case 'range': 
                y_axis = np.arange(num_range_bins) * delta_r
                x_axis = np.arange(num_doppler_bins)
            case _:
                raise ValueError(f"unknown plot_type selected")
        


        if not self.plot3d:
            
            plt.imshow(np.abs(plot_data[:,:,1]).T,
                       extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]],
                       origin='lower', aspect='auto', cmap='viridis')
            plt.colorbar(label="Magnitude")
            plt.ylabel("Range [m]")
            plt.xlabel("Doppler-Bins")
            plt.show()
        else:

            X, Y = np.meshgrid(x_axis, y_axis, indexing='ij')
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, np.abs(plot_data[:,:,1]),
                                   cmap='viridis')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Magnitude")
            ax.set_xlabel("Doppler-Bins")
            ax.set_ylabel("Range [m]")
            ax.set_zlabel("Magnitude")
            plt.show()





