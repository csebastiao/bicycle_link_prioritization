# Bicycle_link_prioritization

Because cycling is an effective solution for making urban transport more sustainable, many cities worldwide are currently announcing extensive bicycle network development plans. These plans typically outline hundreds of kilometers of new links to be built over several years. However, the links in most of these plans are not ordered, and it is unclear how to prioritize them to reach a functional network as fast as possible. This package propose an automatized method that optimally prioritizes all links in a given bicycle network plan. We start from a plan, with potentially an already built network. To go from the initial stage to the final stage, we propose several growth strategy, with options to either keep the network connected, or to proceed in an additive or subtractive order. Greedy optimization can be done on directness and coverage.

## To add as a local package

Based on the [The Good Research Code Handbook](https://goodresearch.dev/setup.html#pip-install-your-package), you should:
* Put yourself on your virtual environment and on the folder, such as /bicycle_link_prioritization
* Run pip install -e .

". indicates that we’re installing the package in the current directory. -e indicates that the package should be editable. That means that if you change the files inside the [source] folder, you don’t need to re-install the package for your changes to be picked up by Python."

## Other useful package

To manipulate simplified network, we use the package [NERDS_OSMnx](https://github.com/anerv/NERDS_osmnx). Based on the simplification function of OSMnx (Boeing, G. 2017. "[OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks](https://geoffboeing.com/publications/osmnx-complex-street-networks/)." *Computers, Environment and Urban Systems* 65, 126-139. doi:10.1016/j.compenvurbsys.2017.05.004) we adapt it to our use to be able to simplify graph while discriminating for an attribute, and add the possibility to transform a NetworkX MultiDiGraph object to a Graph object.
