"""
This script contains parsers for command line arguments so that any
of the three simulations in this package can be run with desired
specifications. Arguments include -s (--sim_type), which specifies
which simulation to run; -p (--poli_affil), which specifices the political
affiliation on which to run the simulation; -f (--frac_data), which specifies
whether to run the simulation on only a fraction of the data; and
-fs and -fe (--frac_start and --frac_end, respectively), which specify
the user fraction start and end percentages as decimals.

The main function then runs the chosen simulation with chosen specs.
"""

from argparse import ArgumentParser
from acrophily_sims.acrophily_sims import ProbDiffSim, MeanAbsDiffSim, AcrophilySim

# Initialize argument parser for command line:
parser = ArgumentParser(prog='Acrophily Simulations')

# Create command for simulation type:
parser.add_argument('-s', '--sim_type', default="acrophily",
                    help="Type of simulation you wish to run (acrophily/prob_diff/mean_abs_diff)")

# Create command for political affiliation:
parser.add_argument('-p', '--poli_affil', default="left",
                    help="Political affiliation subset you wish to run simulation on (left/right)")

# Create commands for using fractions of data:
parser.add_argument('-f', '--frac_data', default=False,
                    help="Whether you wish to run simulation on fraction of data (True/False)", type=bool)
parser.add_argument("-fs", "--frac_start", default=0.0,
                    help="Fraction of users you wish to start from (beginning at 0.0)", type=float)
parser.add_argument("-fe", "--frac_end", default=1.0,
                    help="Fraction of users you wish to end at (ending at 1.0)", type=float)


# Define main function to run simulation with desired specifications from command terminal:
def main(args=None):
    args = parser.parse_args(args=args)

    if args.sim_type == 'mean_abs_diff':
        sim = MeanAbsDiffSim(poli_affil=args.poli_affil, frac_data=args.frac_data,
                             frac_start=args.frac_start, frac_end=args.frac_end)
        sim.main()

    elif args.sim_type == 'prob_diff':
        sim = ProbDiffSim(poli_affil=args.poli_affil, frac_data=args.frac_data,
                          frac_start=args.frac_start, frac_end=args.frac_end)
        sim.main()

    else:
        sim = AcrophilySim(poli_affil=args.poli_affil, frac_data=args.frac_data,
                           frac_start=args.frac_start, frac_end=args.frac_end)
        sim.main()