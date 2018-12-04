# MC code for lithium intercalation in graphite

This code is for a collaboration currently between Mike Mercer, Chao Peng and Denis Kramer. It is based the Python code used Mike and Denis's published work, looking at entropy profiles of lithium insertion in lithium manganese oxide spinel. This is currently an early beta version, having been partially converted to hexagonal geometry. The aim is to replicate Kristin Perrson's phase diagram of bulk lithium insertion in graphite, with the eventual aim to use it to predict surface phase diagrams. See references below.  

## Getting Started

The primary code is "mc_argparse.py" and more detailed instructions are in the header of that script. Basically, the code is set up to be run on a cluster. The current version of the code can either be run interactively, or on a cluster, and is operated by entering command line arguments. An example job submission script: "argparse_sh.sh", is provided. However, there are default arguments in the code such that it will run without any provided arguments.

The main code performs Grand Canonical Monte Carlo simulations and determines thermodynamic averages (occupation, internal energy, partial molar entropy) over a set of input chemical potential values, which are output into a csv file. Generally this has been done as a two step process: a first run which saves equilibrated saved in a serialised Python format (Pickle). This part is run using a single CPU core per submission and is currently the most time consuming. Part two reads in the saved lattice files and calculates fluctuations in occupation and energy. The code is set up so that part two can be run as many serial jobs at the same time. There is a separate script that can aggregate all the statistics from these runs into a single csv files. These files were used in bespoke Python Matplotlib scripts for the publication, but could be plotted in any desired format. 

### Prerequisites

The code is written to be Python 2.7 compatible. The requirements are numpy and pandas modules. There is no makefile and the script can currently be run as is, although some of the paths in the script are set up to the file structure of the Lancaster HEC and so may need to be edited to run interactively or on another cluster. 

### Installing

No installation is currently required. Provided Python and the modules outlined above are installed, it can be run as is.

## Running the tests

There are currently no unit tests in place. The behaviour of the code can be checked by running the default arguments. 

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use git for versioning. For the versions available, see the [tags on this repository](https://github.com/mikemercer87/mc-graphite). 

## Authors

* **Michael Mercer** - *Initial work* - [MikeMercer](https://github.com/mikemercer87/)

## License

This project is part of the Faraday Institution.

## Acknowledgments

Thanks to Sophie Finnigan who contributed to an early version of this code. Thanks to co-workers Harry Hoster, Denis Kramer, Daniel Richards, Chao Peng for input along the way.

## References

M.P. Mercer et al. "The influence of point defects on the entropy profiles of Lithium Ion battery cathodes: a lattice gas Monte Carlo Study" [MercerEntropy](https://www.sciencedirect.com/science/article/pii/S0013468617308836)

K. Perrson et al., "Thermodynamic and kinetic properties of the Li-graphite system from first-principles calculations" [PerrsonGraphite](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.82.125416)

