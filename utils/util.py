import torch
import dnnlib
import legacy
import copy


def load_networks(config, device):

    generator = config["projector_params"]["general"]["use_generator"]
    network_pkl = config["projector_params"]["generators"][generator]["network_pkl"]
    print('Loading G network from "%s"...' % network_pkl, flush=True)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    detector = config["projector_params"]["general"]["use_detector"]
    url = config["projector_params"]["detectors"][detector]["pth"]
    print('Loading D network from "%s"...' % url, flush=True)
    with dnnlib.util.open_url(url) as f:
        D = torch.jit.load(f).eval().to(device)

    return G, D
