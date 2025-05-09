# ML Dependency Injection System

A flexible machine learning framework that uses dependency injection to manage component dependencies.

## Overview

This system demonstrates using dependency injection to create loosely coupled ML components. It features:

- Automatic dependency resolution based on component requirements
- YAML configuration for component selection and parameters
- Extensible design with base classes for each component type
- Simple factory methods to create components with their dependencies

## Components

The system includes the following component types:

- **DataLoader**: Load data from files (CSV, Parquet)
- **Preprocessor**: Normalize or transform data
- **Model**: ML model implementation
- **MetricFunction**: Evaluate model performance
- **Optimizer**: Update model parameters
- **Tracker**: Track metrics and parameters
- **TrainLoop**: Orchestrate the training process

## Example Usage

1. Create a YAML configuration file:

```yaml
dataloader:
  class: CSVDataLoader
  filename: "data/sample_data.csv"

metricfunction:
  class: MSE

tracker:
  class: StdoutTracker

preprocessor:
  class: MinMaxNormalizer

model:
  class: LinearModel
  input_dim: 4

optimizer:
  class: Adam
  learning_rate: 0.01

trainloop:
  class: StandardTrainLoop
  epochs: 50
  batch_size: 16
```

2. Use the factory method to create and run a training loop:

```python
from ml_di_system.components.trainloop import TrainLoop

# Create and execute the training loop
train_loop = TrainLoop.create("configs/sample_config.yaml")
train_loop.execute()
```

3. The system automatically:
   - Creates all component instances
   - Resolves dependencies between components
   - Passes the correct instances to component constructors

## Getting Started

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Generate sample data:

```bash
python -m ml_di_system.data.generate_sample_data
```

3. Run the main script:

```bash
python -m ml_di_system.main --generate-data
```

## Extending the System

To add a new component implementation:

1. Inherit from the appropriate base class
2. Implement required abstract methods
3. Update your configuration to use the new class

Note: No need to modify the dependency injection system itself.