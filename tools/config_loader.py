
import yaml
from dotmap import DotMap


def get_config(config_name='settings'):

    with open('settings/{}.yaml'.format(config_name), 'r') as f:

        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)
    return config
