# I’ve only just started this project, and I’ll continue programming every week. Goal is to create a playground for MIMO radar signal processing

# You can already use the code, but please wait until V1.0, it will be ready soon.

### V0.1 DEV
IQ data generation up to the SIP stage Range plotting is implemented.

### Next TODO for V1.0
- Clean up and comment the existing code --> done
- Expand configuration files and available parameters
- SIP modules: Doppler FFT,CA-CFAR, Doppler plot, MIMO Doppler phase‑compensation, azimuth and elevation estimation
- Implement dynamic pipeline configuration
- Implement a basic GUI for configuration options
- Improve runtime performance

### Planned for V2.0
- Extend the README with explanations and mathematical background
- Improve world and target simulation
- Implement Tx amplitude calculation based on electrical parameters
- Implement Rx amplitude calculation based on time‑of‑flight and target conditions
- Add (root)‑MUSIC estimation module
- Add OS‑CFAR module
- Add GUI with controls and improved visualization

---

# MIMO Radar Simulation

A modular MIMO radar simulation and signal-processing pipeline implemented in Python. The project generates complex IQ data, processes the data through a configurable SIP pipeline.


---
## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```
---
## Project Structure
```bash
MIMO_RADAR/
├─ config/
│  ├─ radar_config.json
│  └─ targets.json
├─ log/
├─ src/ or project root
│  ├─ main.py
│  ├─ radar.py
│  ├─ sip.py
│  ├─ utility.py
│  └─ world.py
├─ requirements.txt
└─ README.md
```
## Usage
### Start the simulation
```
python bring_up.py
```