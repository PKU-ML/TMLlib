import sys
from argparse import Namespace, ArgumentParser
try:
    import yaml
    yaml_available = True
except:
    yaml_available = False


def get_args(TRAINER_CLASS) -> Namespace:
    if yaml_available and len(sys.argv) == 3 and sys.argv[1] == '--yaml':
        with open(sys.argv[2], 'r') as file:
            yaml_data = yaml.safe_load(file)
        args = Namespace(yaml_data)
    else:
        parser = ArgumentParser()
        TRAINER_CLASS.add_argument(parser)
        args = parser.parse_args()
    return args
