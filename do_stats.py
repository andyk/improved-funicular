import sys
import random

from strategies.random import try_random
from strategies.only_go_right import try_only_go_right
from strategies.down_left import try_down_left
from strategies.fixed_action_order import try_fixed_action_order
from strategies.greedy import try_greedy
from strategies.greedy_fixed_order import try_greedy_fixed_order
from strategies.down_left_greedy import try_down_left_greedy
from strategies.max_space_then_greedy import try_max_space_then_greedy
from strategies.lookahead import try_lookahead
from strategies.mcts import try_mcts

from nick_2048 import Nick2048
from andy_adapter import Andy2048


def print_usage(impls, strats):
    print()
    print("Usage:")
    print(f"\tpython {sys.argv[0]} [implementation] [trial_count] [strategy]")
    print()
    print("Example:")
    impl = random.choice(list(impls.keys()))
    strat = random.choice(list(strats.keys()))
    print(f"\tpython {sys.argv[0]} {impl} 100 {strat}")
    print()
    print()

    print(f"[implementation] is one of:")
    for key, val in impls.items():
        try:
            print(f"\t{key} - {val.info}")
        except AttributeError:
            print(f"\t{key} - no description")
    print()

    print(f"[strategy] is one of:")
    for key, val in strats.items():
        try:
            print(f"\t{key} - {val.info}")
        except AttributeError:
            print(f"\t{key} - no description")
    print()


if __name__ == "__main__":
    impls = {
        "nick": Nick2048,
        "andy": Andy2048,
    }
    lookahead_str = "Monte Carlo tree search variant with depth limited to {}."

    def try_lookahead_1(impl, trials):
        try_lookahead(impl, trials, 1)

    try_lookahead_1.info = lookahead_str.format(1)

    def try_lookahead_2(impl, trials):
        try_lookahead(impl, trials, 2)

    try_lookahead_2.info = lookahead_str.format(2)

    def try_lookahead_3(impl, trials):
        try_lookahead(impl, trials, 3)

    try_lookahead_3.info = lookahead_str.format(3)

    def try_lookahead_4(impl, trials):
        try_lookahead(impl, trials, 4)

    try_lookahead_4.info = lookahead_str.format(4)

    def try_lookahead_5(impl, trials):
        try_lookahead(impl, trials, 5)

    try_lookahead_5.info = lookahead_str.format(5)

    strats = {
        "only_go_right": try_only_go_right,
        "random": try_random,
        "down_left": try_down_left,
        "fixed_action_order": try_fixed_action_order,
        "greedy": try_greedy,
        "greedy_fixed_order": try_greedy_fixed_order,
        "down_left_greedy": try_down_left_greedy,
        "max_space_then_greedy": try_max_space_then_greedy,
        "lookahead_1": try_lookahead_1,
        "lookahead_2": try_lookahead_2,
        "lookahead_3": try_lookahead_3,
        "lookahead_4": try_lookahead_4,
        "lookahead_5": try_lookahead_5,
        "mcts": try_mcts,
    }

    if len(sys.argv) != 4:
        print_usage(impls, strats)
        sys.exit(0)

    impl_name = sys.argv[1]
    trial_count = int(sys.argv[2])
    strat_name = sys.argv[3]

    if impl_name not in impls or strat_name not in strats:
        print_usage(impls, strats)
        sys.exit(0)

    print(
        f"\nRunning {trial_count} trials with "
        f"{impl_name}'s impl to test {strat_name}\n"
    )

    implementation = impls[impl_name]
    strategy = strats[strat_name]
    strategy(implementation, trial_count)
