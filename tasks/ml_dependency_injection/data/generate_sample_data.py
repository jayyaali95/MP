import numpy as np
import pandas as pd
import os


def main():
    np.random.seed(42)
    
    # Generate sample data for a linear regression problem
    n_samples = 1000
    n_features = 4
    
    X = np.random.randn(n_samples, n_features)
    
    true_weights = np.array([0.5, -0.3, 0.8, -0.2])
    true_bias = 2.0
    
    y = np.dot(X, true_weights) + true_bias + 0.1 * np.random.randn(n_samples)
    
    column_names = [f'feature_{i}' for i in range(n_features)] + ['target']
    data = np.column_stack([X, y])
    df = pd.DataFrame(data, columns=column_names)
    
    output_path = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Generated sample data with {n_samples} samples and {n_features} features.")
    print(f"Saved to: {output_path}")
    print(f"True weights: {true_weights}")
    print(f"True bias: {true_bias}")


if __name__ == "__main__":
    main() 