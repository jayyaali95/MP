from typing import Type, Any
import pandas as pd


class Column:
    def __init__(self, dtype: Type, *, required: bool = False, min: Any = None, max: Any = None):
        self.dtype = dtype
        self.required = required
        self.min = min
        self.max = max

class DataFrame(pd.DataFrame):
    def __init__(self, *args, schema=None, **kwargs):
        super().__init__(*args, **kwargs)
        print(schema)
        self._schema = schema

    def validate(self):
        if self._schema is None:
            return
        columns = self._schema._columns_
        for k, col in columns.items():
            if k not in self.columns:
                raise ValueError(f"Missing required column: {k}")
            series = self[k]
            if not pd.api.types.is_dtype_equal(series.dtype, pd.Series(dtype=col.dtype).dtype):
                raise TypeError(f"Column '{k}' expected dtype {col.dtype}, got {series.dtype}")
            if col.required and series.isnull().any():
                raise ValueError(f"Column '{k}' contains null values but is required")
            if col.min is not None and (series < col.min).any():
                raise ValueError(f"Column '{k}' has values below minimum {col.min}")
            if col.max is not None and (series > col.max).any():
                raise ValueError(f"Column '{k}' has values above maximum {col.max}")

def validate(func):
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, DataFrame):
                arg.validate()
        for arg in kwargs.values():
            if isinstance(arg, DataFrame):
                arg.validate()
        return func(*args, **kwargs)
    return wrapper

@validate
def describe_df(df: DataFrame) -> None:
    print(df.describe())
