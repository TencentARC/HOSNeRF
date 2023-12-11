import imp

def create_optimizer(cfg, network):
    module = cfg.optimizer_module
    optimizer_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, optimizer_path).get_optimizer(cfg, network)
