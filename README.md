# Bipedal Walker Training with Genetic Algorithms

## Overview

This project contains various scripts that implement evolutionary strategies and genetic algorithms to train models in the "BipedalWalker-v3" environment from OpenAI's Gym. The core of the project uses Keras models for neural networks and DEAP for genetic algorithm optimization. The focus is on optimizing the weights of neural networks for controlling the walker agent in a reinforcement learning environment.

### Key Features:
- **Custom Neural Network Models:** Defined using Keras and TensorFlow.
- **Genetic Algorithm (GA) Optimization:** Implemented using DEAP to evolve model weights.
- **BipedalWalker-v3 Simulation:** Agents are trained to walk in this environment.
- **Logging & Model Saving:** Models and training logs are saved using pickle for future reference.

## Files Description

1. **auto_train_velocity.py:** 
   - Trains a neural network model for BipedalWalker using velocity-based metrics for optimization.
   - Includes custom weight transformation functions for converting model weights between vector and matrix forms.

2. **auto_train_firstPlus.py:**
   - Implements a variation of training with an alternative genetic algorithm approach, possibly focusing on improving the initial selection or exploration phase.

3. **auto_train_PlusWPenalty.py:**
   - Incorporates penalties (likely for failing or inefficient behavior) to improve the model’s performance in the BipedalWalker environment.

4. **auto_train_FilteredGA.py:**
   - Uses a filtered version of a genetic algorithm, potentially limiting certain types of individuals or mutations to improve performance.

5. **auto_train.py:**
   - Main training script that sets up the environment, defines the neural network model, and handles the entire evolutionary process, including evaluation and logging.

## Installation

To set up and run the project locally:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have Gym's `BipedalWalker-v3` environment installed:
   ```bash
   pip install gym[Box2D]
   ```

4. (Optional) Install TensorFlow if not already installed:
   ```bash
   pip install tensorflow
   ```

## Usage

To train a model, run one of the provided scripts. For example:
```bash
python auto_train.py
```

The script will run the genetic algorithm optimization on the neural network model and print training logs to the console. The final model and training logs will be saved as `.pkl` files in the current directory.

## Customization

- **Population size, generations, and mutation rate** can be customized in each script:
  - `populationControl` sets the number of individuals in the population.
  - `generations` determines how many generations the algorithm will run.
  - `mutationRate` controls the mutation probability.

- **Neural Network Architecture**: You can modify the neural network model by altering the `model_build()` function in the respective scripts.

## Results

After training, the best individual model and logs are saved in pickle files with a unique filename, based on the "BipedalWalker" task. You can load and analyze these files using `pickle` for further evaluation.

## Contributing

Feel free to contribute by creating pull requests or reporting issues.

## License

This project is licensed under the MIT License.

---

This README provides a high-level overview and instructions for usage, but you may want to customize it based on additional specifics of your project. Let me know if you'd like to add or change anything!
