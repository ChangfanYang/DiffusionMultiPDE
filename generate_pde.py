import yaml
from argparse import ArgumentParser
from scripts import generate_burgers, generate_darcy, generate_poisson, generate_helmholtz, generate_ns_nonbounded, generate_ns_bounded, generate_TE_heat_validate, generate_TE_heat, generate_TE_heat_sigl, generate_TE_heat_sigl_onlyE, generate_NS_heat, generate_MHD, generate_E_flow, generate_VA, generate_Elder
import sys

if __name__ == "__main__":
    arg = sys.argv  # 接收参数

    parser = ArgumentParser(description='Generate PDE file')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True)
    options = parser.parse_args()


    config_path = options.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    name = config['data']['name']
    if name == 'Burgers':
        print('Solving Burgers equation...')
        generate_burgers(config)
    elif name == 'Darcy':
        print('Solving Darcy Flow equation...')
        generate_darcy(config)
    elif name == 'Poisson':
        print('Solving Poisson equation...')
        generate_poisson(config)
    elif name == 'Helmholtz':
        print('Solving Helmholtz equation...')
        generate_helmholtz(config)
    elif name == 'NS-NonBounded':
        print('Solving non-bounded NS equation...')
        generate_ns_nonbounded(config)
    elif name == 'NS-Bounded':
        print('Solving bounded NS equation...')
        generate_ns_bounded(config)
    elif name == 'TE_heat':
        print('Solving TE_heat equation...')
        # generate_TE_heat(config, options.start, options.end)
        # generate_TE_heat_sigl(config, options.start, options.end)
        generate_TE_heat_sigl_onlyE(config, options.start, options.end)
        # generate_TE_heat(config)
        # generate_TE_heat_validate(config)

    elif name == 'NS_heat':
        print('Solving NS_heat equation...')
        generate_NS_heat(config, options.start, options.end)
        # generate_NS_heat(config)

    elif name == 'MHD':
        print('Solving MHD equation...')
        generate_MHD(config, options.start, options.end)
        # generate_MHD(config)

    elif name == 'E_flow':
        print('Solving E_flow equation...')
        generate_E_flow(config, options.start, options.end)
        # generate_E_flow(config)

    elif name == 'VA':
        print('Solving VA equation...')
        generate_VA(config, options.start, options.end)
        # generate_VA(config)
        
    elif name == 'Elder':
        print('Solving Elder equation...')
        generate_Elder(config, options.start, options.end)
        # generate_Elder(config)


    else:
        print('PDE not found')
        exit(1)