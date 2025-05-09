import time
import logging
from contextlib import contextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

@contextmanager
def profiler():
    """Context manager to profile execution time of a code block."""
    start_time = time.perf_counter()
    yield
    elapsed_time = time.perf_counter() - start_time
    logging.info(f"Execution time: {elapsed_time:.6f} seconds")

def add(a: float, b: float) -> float:
    return a + b


@profiler()
def sub(a: float, b: float) -> float:
    return a - b

if __name__ == "__main__":
    # Profile the add function
    with profiler():
        add_result = add(1, 2)
    logging.info(f"Addition result using profiler: {add_result}")

    # Profile the sub function
    sub_result = sub(3, 1)
    logging.info(f"Subtraction result using decorator: {sub_result}")

