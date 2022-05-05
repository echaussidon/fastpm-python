import argparse


def collect_argparser():
    parser = argparse.ArgumentParser(description="From the list of Nodes used in the job, build the associated \
                                                  hostfile with only rank 0 in the first node")

    parser.add_argument('-i', "--input", type=str, required=True,
                        help="txt file with the NodeNames of the job")
    parser.add_argument('-o', "--output", type=str, required=True,
                        help="hostfile which will be used with --distribution=arbitrary, \
                              the rank 0 is alone in its node and the other ranks are --ntask_per_nodes in the other nodes.")
    parser.add_argument("--ntask_per_nodes", type=int, required=False, default=68,
                        help="Number of tasks in each node (except in the first node --> only rank 0)")

    return parser.parse_args()


if __name__ == '__main__':
    args = collect_argparser()

    lines = open(args.input, "r").read().splitlines()

    hostfile = open(args.output, "x")
    for idx, line in enumerate(lines):
        if idx == 0:
            hostfile.write(line + '\n')
        else:
            for _ in range(args.ntask_per_nodes):
                hostfile.write(line + '\n')
    hostfile.close()
