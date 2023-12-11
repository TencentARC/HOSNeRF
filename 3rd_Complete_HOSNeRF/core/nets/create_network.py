import imp

def _query_network(cfg):
    module = cfg.network_module
    module_path = module.replace(".", "/") + ".py"
    network = imp.load_source(module, module_path).Network
    return network


def create_network(cfg):
    network = _query_network(cfg)
    network = network(cfg)
    return network
