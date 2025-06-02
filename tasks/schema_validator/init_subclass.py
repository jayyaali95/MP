from utils import Column

class InitSubclassSchema:
    _columns_: dict = {}

    def __init_subclass__(cls):
        columns = {}
        for k, v in cls.__dict__.items():
            if isinstance(v, Column):
                columns[k] = v
        cls._columns_ = columns
