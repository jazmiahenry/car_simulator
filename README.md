# Car Simulator

## Overview

This package implements a reinforcement learning environment for simulating and optimizing car driving behavior. It includes a customizable car environment, RL agents, and a visualization tool to observe the learning process.

## Features

- Customizable car driving environment
- Integration with Stable Baselines3 for RL algorithms
- Support for multiple cars in the simulation
- Dynamic environment factors (time of day, weather, weekday/weekend)
- Pygame-based visualization of the learning process

## Installation

### Prerequisites

- Python 3.7+
- pip

### Steps

1. Clone the repository:
   ```
   git clone https://github.com/your-username/car-rl-package.git
   cd car-rl-package
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training a model

To train a new RL model, run the `main.py` script with desired parameters:

```
python main.py --num_cars 3 --time_of_day 14:30 --is_rainy --train_steps 20000 --visualize_episodes 5
```

Arguments:
- `--num_cars`: Number of cars in the simulation (default: 3)
- `--time_of_day`: Initial time of day in HH:MM format (default: "08:00")
- `--is_rainy`: Flag to set initial weather to rainy
- `--is_weekday`: Flag to set initial day to weekday
- `--train_steps`: Number of training steps (default: 10000)
- `--visualize_episodes`: Number of episodes to visualize after training (default: 5)
- `--load_model`: Path to a pre-trained model to load (optional)

### Visualizing the environment

The visualization will automatically run after training. It shows the cars moving in the environment, with info boxes displaying current stats and actions for each car.

Controls during visualization:
- `T`: Advance time by 1 hour
- `R`: Toggle rain on/off
- `W`: Toggle weekday/weekend
- Close the window to end the visualization

## Project Structure

- `main.py`: Entry point for training and visualization
- `CarSimulator/`
  - `car_rl_environment.py`: Defines the CarRLEnvironment class
  - `carviz.py`: Implements the CarVisualization class
- `requirements.txt`: List of Python dependencies

## Customization

You can customize the environment by modifying the `CarRLEnvironment` class in `car_rl_environment.py`. This includes changing the reward function, adjusting physics parameters, or adding new features to the simulation.

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your fork
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL algorithms
- [Pygame](https://www.pygame.org/) for visualization

## Contact

For any questions or feedback, please open an issue on the GitHub repository.
