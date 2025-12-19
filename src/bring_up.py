

import logging
from world import WorldSimulation
from utility import setup_logging



if __name__ == "__main__":
    setup_logging(logging.INFO,file_output=True, console_output=True)
    log = logging.getLogger(__name__)
    
    world = WorldSimulation()
    world.run_simulation()
    