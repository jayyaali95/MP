import datetime
import logging
import functools

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def profiler(function):
    """
    A decorator to profile the execution time of a function.
    
    Args:
        function: The function to be profiled.
        
    Returns:
        wrapper: A function that wraps the original function and measures its execution time.
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = function(*args, **kwargs)
        execution_time = datetime.datetime.now() - start_time
        logging.info(f"Function '{function.__name__}' executed in {execution_time}")
        return result
    
    return wrapper

@profiler
def add(a, b):
    return a + b

if __name__ == "__main__":
    result = add(1, 2)
    logging.info(f"Result: {result}")
