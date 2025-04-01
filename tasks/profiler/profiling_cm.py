import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class Profiler:
    """Context manager to profile execution time of a code block."""
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop timing and log the execution time."""
        elapsed_time = time.perf_counter() - self.start_time
        logging.info(f"Execution time: {elapsed_time:.6f} seconds")

def add(a: float, b: float) -> float:
    return a + b

if __name__ == "__main__":
    with Profiler():
        result = add(1, 2)
    logging.info(f"Result: {result}")
    
