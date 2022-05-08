from argparse import ArgumentParser
from .sims_data_prep import SimDataProcessor
from .sims_module import ProbDiffSim, MeanAbsDiffSim

parser = ArgumentParser(prog='Acrophily Simulations')
parser.add_argument('-s', '--sim_type', default="acrophily", help="Type of simulation you wish to run (acrophily/prob_diff/mean_abs_diff/NA)")
parser.add_argument('-o', '--orient', default="left", help="User political orientation you wish to run simulation on (left/right)")
parser.add_argument('-mo', '--merge_only', default=False, help="Meant for merging data instead of running simulation (True/False)", type=bool)
parser.add_argument('-f', '--frac_data', default=False, help="Whether you wish to run simulation on fraction of data (True/False)", type=bool)
parser.add_argument("-fs", "--frac_start", default=0.0, help="Fraction of users you wish to start from (beginning at 0.0)", type=float)
parser.add_argument("-fe", "--frac_end", default=1.0, help="Fraction of users you wish to end at (ending at 1.0)", type=float)

def main(args=None):
    args = parser.parse_args(args=args)

    if args.sim_type == 'prob_diff':

        if args.merge_only:
            data_merger = SimDataProcessor(sim_type=args.sim_type, orient=args.orient)
            data_merger.run()

        else:
            if args.frac_data:
                sim = ProbDiffSim(orient=args.orient, frac_data=args.frac_data,
                              frac_start=args.frac_start, frac_end=args.frac_end)
                sim.run()
            else:
                sim = ProbDiffSim(orient=args.orient)
                sim.run()

    elif args.sim_type == 'mean_abs_diff':
        sim = MeanAbsDiffSim(orient=args.orient, frac_data=args.frac_data, frac_start=args.frac_start,
                             frac_end=args.frac_end)
        sim.run()
