from reddit import reddit, network_anl as reddit_anl
from karate import karate, network_anl as karate_anl

# this function is kept in a separate file to avoid circular imports


def import_network(network):
    """
    Import network data and analysis function based on network name

    Parameters:
    network: 'reddit' or 'karate'
    """

    if network == 'reddit':
        G, s, n = reddit()
        network_anl = reddit_anl
    elif network == 'karate':
        G, s, n = karate()
        network_anl = karate_anl
    return G, s, n, network_anl
