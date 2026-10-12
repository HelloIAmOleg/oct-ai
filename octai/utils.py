import json
from copy import deepcopy

def prepare_config(self, config):
    if config is None:
        config = {}
    default_config = self.default_config
    
    for key in config:
        if key not in default_config:
            raise KeyError(f"Unknown config key: {key}")
        
    new_config = {}
    for k,v in default_config.items():
        new_config[k] = config.get(k, v)

    return deepcopy(new_config)

def read_json(filepath):
    with open(filepath, 'rt') as fp:
        return json.load(fp)
    
def dump_json(obj, filepath):
    with open(filepath, 'wt') as fp:
        json.dump(obj, fp)

if __name__ == "__main__":
    class Foo:
        default_config = {"a": 123, "b": 45, "c": 228}

    config = {"a": 20, "b": 14, "d": 100}

    print(prepare_config(Foo(), config))

    