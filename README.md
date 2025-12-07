# Snake Robot Kinodynamic Planner

This repository contains a Python implementation of a Kinodynamic RRT algorithm designed for a modular snake robot with active track modules. Unlike standard geometric planners that generate "teleporting" paths with sharp corners, this planner utilizes arc-based motion primitives to ensure all trajectories are smooth and drivable by differential-drive tracks.

Key Features:

Kinodynamic RRT: Grows a path tree using physically valid steering inputs (Straight, Left Arc, Right Arc) rather than coordinate sampling.

Follow-the-Leader Simulation: Explicitly simulates the trailing body segments to ensure the entire robot fits through narrow gaps without collision.

Debris Environment: Includes a "Hardcore Maze" generation script to stress-test the planner in simulated disaster scenarios.

Kinematic Constraints: Enforces minimum turning radii to prevent servo stall and track slippage.
