import sys
from argparse import Namespace, ArgumentParser
import json
try:
    import yaml
    yaml_available = True
except:
    yaml_available = False


def get_args(PARAM_CLASS) -> Namespace:
    if len(sys.argv) == 3 and sys.argv[1] == '--yaml':
        if not yaml_available:
            raise ImportError("pyyaml is not installed")
        with open(sys.argv[2], 'r') as file:
            yaml_data = yaml.safe_load(file)
        args = Namespace(yaml_data)
    elif len(sys.argv) == 3 and sys.argv[1] == '--json':
        with open(sys.argv[2], 'r') as file:
            json_data = json.load(file)
        args = Namespace(json_data)
    else:
        parser = ArgumentParser()
        PARAM_CLASS.add_argument(parser)
        args = parser.parse_args()
    return args
