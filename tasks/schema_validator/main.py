from utils import DataFrame, Column, describe_df
from metaclass import Schema as MetaClassSchema
from init_subclass import InitSubclassSchema

class TestSchemaInitSubclassBase(InitSubclassSchema):
    days = Column(int, min=0, max=10)
    probability = Column(float, required=True)
    feature = Column(str)
    _allow_extra_columns = True

class TestSchemaMetaClassBase(MetaClassSchema):
    days = Column(int, min=0, max=10)
    probability = Column(float, required=True)
    feature = Column(str)
    _allow_extra_columns = False

meta_class_based_df = DataFrame({'days': [1,2,3], 'probability': [0.1, 0.5, 0.9], 'feature': ['a','b','c'], 'extra': [1,2,3]}, schema=TestSchemaMetaClassBase)
init_subclass_based_df = DataFrame({'days': [1,2,3], 'probability': [0.1, 0.5, 0.9], 'feature': ['a','b','c'], 'extra': [1,2,3]}, schema=TestSchemaInitSubclassBase)


describe_df(meta_class_based_df)
describe_df(init_subclass_based_df)
