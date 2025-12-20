
from radar import Radar, RadarFrontend
from dataclasses import dataclass, field

import logging
import os
import json
import numpy as np


class WorldSimulation:
    """
    High‑level simulation environment for radar–target interaction.

    The `WorldSimulation` class initializes a radar instance, loads target
    definitions from configuration, computes all target trajectories and
    geometric relations, and finally triggers the radar frontend to generate
    a simulated data cube.

    Parameters
    ----------
    None

    Attributes
    ----------
    radar : Radar
        Radar system containing frontend, backend, and configuration.
    sim_time_res : int
        Total number of time steps (num_chirps × num_samples_per_chirp).
    targets : list of Target
        List of simulated targets loaded from configuration.

    Notes
    -----
    The simulation workflow is:
        1. Load target definitions from JSON.
        2. Compute per‑sample target positions and distances.
        3. Log diagnostic information.
        4. Trigger radar signal generation via `radar.power_on()`.

    Methods
    -------
    place_targtes()
        Load and construct all target objects from JSON configuration.
    update_world()
        Compute target trajectories and geometric relations.
    run_simulation()
        Execute the full radar simulation.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.radar = Radar()
        self.sim_time_res = self.radar.radar_config.num_chirps * self.radar.radar_config.num_samples_per_chirp
        self.targets = self.place_targtes()

        self.update_world()

    def place_targtes(self):
        tgt_config_path = os.path.join(os.path.dirname(__file__),"..","config","targets.json")
        with open(tgt_config_path) as jf:
            tgt_config = json.load(jf)
        targets = [Target.from_dict(t,self.sim_time_res,self.radar.radar_config.num_tx_antenna,self.radar.radar_config.num_rx_antenna) for t in tgt_config["targets"]]
        return targets
    
    def update_world(self):
        for tgt in self.targets:
            tgt.update_target_data(self.radar.frontend)
            tgt._debug_target(10)
            self.logger.info(f"{tgt}")
   


    def run_simulation(self):
        self.radar.power_on(self.targets)


        


@dataclass
class Target:
    """
    Representation of a moving radar target in 3D space.

    A `Target` stores its initial position, velocity vector, and all derived
    quantities required for radar signal simulation, including per‑sample
    world positions, distances to TX/RX antennas, Doppler velocity, azimuth,
    and elevation.

    Parameters
    ----------
    name : str
        Identifier of the target.
    init_world_pos : ndarray, shape (3,)
        Initial 3D position in world coordinates [m].
    vel_vec : ndarray, shape (3,)
        Constant velocity vector [m/s].
    sim_time_res : int
        Number of simulation time steps.
    num_rx : int
        Number of receive antennas.
    num_tx : int
        Number of transmit antennas.

    Attributes
    ----------
    cur_world_pos : ndarray, shape (sim_time_res, 3)
        Target position at each time step.
    distance_to_radar_m : ndarray
        Distance from target to radar origin at each time step.
    distance_to_tx : ndarray, shape (sim_time_res, num_tx)
        Distance to each TX antenna.
    distance_to_rx : ndarray, shape (sim_time_res, num_rx)
        Distance to each RX antenna.
    rel_velocity_m_s : ndarray
        Radial velocity relative to radar line‑of‑sight.
    azimuth_deg : ndarray
        Azimuth angle in degrees.
    elevation_deg : ndarray
        Elevation angle in degrees.
    """
    logger = logging.getLogger(__name__) 
    name: str = ""
    init_world_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel_vec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    cur_world_pos: np.ndarray = field(init=False)
    distance_to_radar_m:  np.ndarray = field(default_factory=lambda: np.array([]))
    distance_to_tx: np.ndarray = field(default_factory=lambda: np.array([]))
    distance_to_rx: np.ndarray = field(default_factory=lambda: np.array([]))
    rel_velocity_m_s: np.ndarray = field(init=False)
    azimuth_deg: np.ndarray = field(init=False)
    elevation_deg: np.ndarray = field(init=False)

    sim_time_res: int = 0   
    num_rx : int = 0
    num_tx : int = 0

    def __post_init__(self):
        self.cur_world_pos = np.zeros((self.sim_time_res, 3))
        self.distance_to_radar_m = np.zeros(self.sim_time_res)
        self.distance_to_tx = np.zeros((self.sim_time_res,self.num_tx))
        self.distance_to_rx = np.zeros((self.sim_time_res,self.num_rx))
        self.rel_velocity_m_s = np.zeros(self.sim_time_res)
        self.azimuth_deg = np.zeros(self.sim_time_res)
        self.elevation_deg = np.zeros(self.sim_time_res)
    
    def __repr__(self):
        return (
            f"Target(name={self.name})\n"
            f"  init_world_pos: {self.init_world_pos}\n"
            f"  vel_vec: {self.vel_vec}\n"
            f"  distance_to_radar_m[0]: {self.distance_to_radar_m[0] if self.distance_to_radar_m.size else 'n/a'}\n"
            f"  rel_velocity_m_s[0]: {self.rel_velocity_m_s[0] if self.rel_velocity_m_s.size else 'n/a'}\n"
            f"  azimuth_deg[0]: {self.azimuth_deg[0] if self.azimuth_deg.size else 'n/a'}\n"
            f"  elevation_deg[0]: {self.elevation_deg[0] if self.elevation_deg.size else 'n/a'}\n"
        )

    @classmethod
    def from_dict(cls, d: dict, sim_time_res: int = 0, num_tx: int = 0, num_rx: int = 0):
        pos = np.array([d["init_world_pos"]["x"],
                        d["init_world_pos"]["y"],
                        d["init_world_pos"]["z"]], dtype=float)
        v_vec = np.array([d["vel_vec"]["x"],
                          d["vel_vec"]["y"],
                          d["vel_vec"]["z"]], dtype=float)
        return cls(
            name=d["name"],
            init_world_pos=pos,
            vel_vec=v_vec,
            sim_time_res=sim_time_res,
            num_tx = num_tx,
            num_rx = num_rx
        )

    
    def update_target_data(self,radar_frontend :RadarFrontend):
        t = np.arange(self.sim_time_res) * radar_frontend.dt
        self.cur_world_pos = self.init_world_pos[None, :] + t[:, None] * self.vel_vec[None, :]
        
        rel_pos = self.cur_world_pos - radar_frontend.radar_pos[None, :]

        self.distance_to_radar_m = np.linalg.norm(rel_pos, axis=1)
        
        tx_positions = np.array([radar_frontend.radar_pos + tx.rel_pos_to_frontend
                                 for tx in radar_frontend.tx_antennas])
        rx_positions = np.array([radar_frontend.radar_pos + rx.rel_pos_to_frontend
                                 for rx in radar_frontend.rx_antennas])
        
        self.distance_to_tx = np.linalg.norm(self.cur_world_pos[:, None, :] - tx_positions[None, :, :], axis=2)
        self.distance_to_rx = np.linalg.norm(self.cur_world_pos[:, None, :] - rx_positions[None, :, :], axis=2)
        
        self.rel_velocity_m_s = np.dot(rel_pos, self.vel_vec) / np.linalg.norm(rel_pos, axis=1)
        
        self.azimuth_deg = np.degrees(np.arctan2(rel_pos[:, 1], rel_pos[:, 0]))
        self.elevation_deg = np.degrees(np.arctan2(rel_pos[:, 2], np.linalg.norm(rel_pos[:, :2], axis=1)))


    def _debug_target(self, n=5):
        self.logger.debug("=== Debug Target Data ===")
        self.logger.debug(f"Initial position:{self.init_world_pos}")
        self.logger.debug(f"Velocity vector:{ self.vel_vec}")
        self.logger.debug(f"First positions: {self.cur_world_pos[:n]}")
        self.logger.debug(f"First distances to radar: {self.distance_to_radar_m[:n]}")
        self.logger.debug(f"First rel velocities:{self.rel_velocity_m_s[:n]}")
        self.logger.debug(f"First azimuths: {self.azimuth_deg[:n]}")
        self.logger.debug(f"First elevations:{ self.elevation_deg[:n]}")