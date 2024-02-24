import yaml
from argparse import Namespace

def yamlparser(yaml_file):

    with open(yaml_file) as file:
        data = yaml.safe_load(file)

    args = Namespace(data)

    return args
