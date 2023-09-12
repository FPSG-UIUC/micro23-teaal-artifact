from ruamel.yaml import YAML

def dump(data, fn):
    yaml = YAML(typ='safe', pure=True)
    with open(fn, "wb") as f:
        yaml.dump(data, f)

def load(fn):
    yaml = YAML(typ='safe', pure=True)
    with open(fn, 'r') as stream:
        data_loaded = yaml.load(stream)
    return data_loaded

