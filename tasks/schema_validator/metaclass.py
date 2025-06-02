from utils import Column


class SchemaMeta(type):
    def __new__(mcs, name, bases, namespace):
        cols = {}
        for key, value in list(namespace.items()):
            if isinstance(value, Column):
                cols[key] = value
        namespace['_columns_'] = cols
        return super().__new__(mcs, name, bases, namespace)
    
Schema = SchemaMeta('Schema', (), {})
