# Third-Year-Project

## Running the project
In order to run this project, you will need to have installed SUMO (https://sumo.dlr.de/docs/Installing/index.html), ensuring that it is in your environment variables.

Next, you will need to install the necessary libraries with the help of the "requirements.txt" file.

Finally, you can move into the "src" folder and run "python main.py" which will start training the example agent.

## Project structure
 - agents: contains a variety of RL and non-RL agents
 - envs: contains a variety of environments that control states and rewards
 - networks: contains a variety of road network structures
 - routes: contains a variety of vehicle flow types
 - saved: stores models and data during training
 - sensors: contains lane area detectors for road networks
 - sumo: folder in which the environment will load all sumo related files (modification is not recommended)
 - utils: contains a variety of helper classes
 