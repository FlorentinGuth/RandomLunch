""" Random lunch script.

Before the random lunch:
Running this script without arguments will propose a configuration for the next random lunch, based on the latest
participants CSV file. The participants and configurations are saved and identified with their date.

After the random lunch:
Running this script with the total cost and the payer team will record the configuration to the history and do the
bookkeeping. The configuration file previously created should reflect the actual configuration that took place and can
be easily edited to account for last-minute changes.
"""

import argparse
import json
import random
import datetime
from pathlib import Path
import shutil
import tqdm
from typing import NamedTuple
import pandas as pd
import scipy.special


email_field = "Your email "
name_field = "Your first name and last name"
team_field = "Your team supervisor"

random_lunch_day = 1  # Tuesday

ideal_group_size = 4  # Try to make groups of this size.


class History(NamedTuple):
    """ A history records information about past random lunches:
    - the number of times each pair of participants has been at the same table,
    - the sum paid by each team,
    - the true cost of random lunches for each team.
    """
    pairs: dict  # sorted pair of email addresses -> number of times they have been at the same table
    paid: dict  # team name -> sum paid for all past random lunches
    cost: dict  # team name -> sum of cost for all past random lunches


def get_cost(configuration, participants, history):
    """ Measure the cost of a configuration according to past history of random lunches.
    :param configuration: list of sorted lists of integers, corresponding to participant indices.
    :param participants: DataFrame sorted by email addresses.
    :param history: dictionary of sorted email address pairs to number of past lunches.
    :return: the cost of the configuration (an integer), the smaller the better.
    """
    cost = 0
    for pair in enumerate_pairs(configuration):
        p0, p1 = participants.iloc[pair[0]], participants.iloc[pair[1]]
        emails = str(tuple(sorted((p0[email_field], p1[email_field]))))
        num_times = history.get(emails, 0)
        if p0[team_field] == p1[team_field]:
            num_times += 1000
        cost += num_times ** 2
    return cost


def update_history(configuration, participants, cost, payer_team, history):
    """ Updates in-place the history to account for the chosen configuration. """
    for pair in enumerate_pairs(configuration):
        # Record the pair in the history.
        emails = str(tuple(sorted((participants.loc[participants.index[pair[0]], email_field],
                                   participants.loc[participants.index[pair[1]], email_field]))))
        history.pairs[emails] = history.pairs.get(emails, 0) + 1

    history.paid[payer_team] = history.paid.get(payer_team, 0) + cost

    num_participants = len(participants)
    for i in range(num_participants):
        team = participants.iloc[i][team_field]
        history.cost[team] = history.cost.get(team, 0) + cost / num_participants


def enumerate_pairs(configuration):
    """ Returns sorted pairs of participants which are in a same group. """
    for group in configuration:
        for pair in enumerate_groups(num_participants=len(group), group_size=2):
            # We are guaranteed that pairs are sorted, as both group and pair and increasing lists.
            yield tuple([group[participant] for participant in pair])


# Enumeration of configurations.


def argmax(iterable, score):
    """ Finds the first item in iterable with the highest score. """
    best_item = None
    best_score = float("-inf")
    for item in iterable:
        if (item_score := score(item)) > best_score:
            best_item = item
            best_score = item_score
    return best_item


def get_group_sizes(num_participants):
    """ Break the number of participants into group sizes.
    We try to make groups that are of size ideal_group_size or ideal_group_size +/- 1.
    :return: a dictionary whose keys are group sizes and whose values are the number of such groups.
    """
    # Start with k groups of size group_size and a smaller rest.
    num_full_groups = num_participants // ideal_group_size
    rest = num_participants % ideal_group_size

    # If there are not enough participants, then we can only make one group.
    if num_full_groups == 0:
        groups = {rest: 1}

    # If the rest is empty, we are done, there are only groups of the ideal size.
    elif rest == 0:
        groups = {ideal_group_size: num_full_groups}

    # Otherwise, we try to make groups of size ideal_group_size - 1 or ideal_group_size + 1, if possible.
    # This is done by moving participants from full groups to the rest groups, or by moving participants from the rest
    # group to full groups.
    else:
        # With ideal_group_size - 1: may be impossible if there are not enough full_groups.
        num_participants_to_move = ideal_group_size - 1 - rest
        small_groups = {ideal_group_size: num_full_groups - num_participants_to_move,
                        ideal_group_size - 1: num_participants_to_move + 1}

        # With larger groups: always possible.
        additional_participants_per_group = rest // num_full_groups  # Should be zero most of the time.
        rest_rest = rest % num_full_groups
        big_groups = {ideal_group_size + additional_participants_per_group: num_full_groups - rest_rest,
                      ideal_group_size + additional_participants_per_group + 1: rest_rest}

        # Filter impossible configurations and select the one with the most groups of ideal size.
        configurations = [groups for groups in [small_groups, big_groups]
                          if all(num_groups >= 0 for num_groups in groups.values())]
        groups = argmax(configurations, lambda groups: groups.get(ideal_group_size, 0))

    # Filter zeroes.
    return {group_size: num_groups for group_size, num_groups in groups.items() if num_groups > 0}


def test_group_sizes():
    """ Small function to test computed group sizes. """
    for num_participants in range(1, 31):
        groups = get_group_sizes(num_participants)
        assert sum(group_size * num_groups for group_size, num_groups in groups.items()) == num_participants
        print(num_participants, groups)


def enumerate_groups(num_participants, group_size):
    """ Returns an enumerator over all possible groups of size group_size. """
    if group_size == 0:
        # Base case: there is one group of size zero.
        yield []
    else:
        # We enumerate over first participants of the group and do a recursive call.
        for first_participant in range(num_participants):
            for group in enumerate_groups(num_participants - first_participant - 1, group_size - 1):
                yield [first_participant] + [participant + first_participant + 1 for participant in group]


def test_enumerate_groups():
    """ Small function to test enumeration of all possible groups. """
    from scipy.special import binom
    num_participants = 4
    for group_size in range(num_participants + 2):
        groups = list(enumerate_groups(num_participants, group_size))
        assert all(len(group) == group_size for group in groups)
        assert len(groups) == binom(num_participants, group_size)
        print(num_participants, group_size, groups)


def enumerate_configurations(num_participants, group_sizes):
    """ Returns an enumerator over all possible configurations.
    Participants are represented here as an integer from 0 to num_participants - 1.
    :return: enumerator over configurations, which are a list of groups, and each group is a list of participants.
    """
    # Base case: zero participants.
    if num_participants == 0:
        yield []
    else:
        # We start by enumerating possible groups with participant 0.
        for group_size_0 in group_sizes:
            # Participant zero is in a group of size group_size_0.
            group_sizes_0 = group_sizes.copy()
            group_sizes_0[group_size_0] -= 1
            if group_sizes_0[group_size_0] == 0:
                del group_sizes_0[group_size_0]

            # Enumerate over groups containing participant 0.
            for group_0 in enumerate_groups(num_participants - 1, group_size_0 - 1):
                # Handle the translation of participant numbers.
                group_0 = [0] + [participant + 1 for participant in group_0]
                translation_table = {}
                num_skipped = 0
                for participant in range(num_participants):
                    if participant in group_0:
                        num_skipped += 1
                    else:
                        translation_table[participant - num_skipped] = participant

                # Do a recursive call to enumerate over ways to partition the rest:
                for configuration in enumerate_configurations(num_participants - group_size_0, group_sizes_0):
                    yield [group_0] + [[translation_table[participant] for participant in group]
                                       for group in configuration]


def num_configurations(num_participants, group_sizes):
    """ Compute the number of configurations for a given partition into groups. """
    num = 1
    div = 1
    for group_size, num_groups in group_sizes.items():
        for _ in range(num_groups):
            num *= scipy.special.comb(num_participants, group_size)
            num_participants -= group_size
        div *= scipy.special.factorial(num_groups)
    return num // div


def test_enumerate_configs():
    """ Small function to test the enumeration of configurations. """
    num_participants = 7
    group_sizes = {2: 2, 3: 1}
    configurations = list(enumerate_configurations(num_participants, group_sizes))
    assert len(configurations) == num_configurations(num_participants, group_sizes)
    for configuration in configurations:
        assert sorted(sum(configuration, start=[])) == list(range(num_participants))
        print(configuration)


def get_random_minimal_cost_configuration(participants, group_sizes, history):
    """ Returns a random configuration among those of minimal cost. """
    num_participants = len(participants)

    # Compute all configuration of with minimal cost.
    best_cost = float("inf")
    best_configurations = []
    try:
        iterator = tqdm.tqdm(enumerate_configurations(num_participants, group_sizes),
                             total=num_configurations(num_participants, group_sizes))
        for configuration in iterator:
            # Translate participants from numbers to email addresses.
            cost = get_cost(configuration, participants, history)

            if cost < best_cost:
                best_cost = cost
                best_configurations = [configuration]
            elif cost == best_cost:
                best_configurations.append(configuration)
            iterator.desc = f"Found {len(best_configurations)} configurations of cost {best_cost}"
    except KeyboardInterrupt:
        # Impatient user, just return one of the best configurations found so far
        pass

    # Return a random configuration among all best ones.
    return best_configurations[random.randrange(0, len(best_configurations))]


def get_participants(participants_file):
    """ Returns the list of today's participants, as a DataFrame in random order. """
    df = pd.read_csv(participants_file)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame.
    return df


def save_configuration(configuration, participants, participants_file, random_lunch_date):
    """ Print and save the chosen configuration, and copy the participants file there. """
    lines = []
    for i, group in enumerate(configuration, start=1):
        lines.append(f"Table {i}:")
        for participant in group:
            lines.append(participants.iloc[participant][name_field])
        lines.append("")
    configuration_str = "\n".join(lines)

    print(configuration_str)

    folder = Path("random_lunches")
    folder.mkdir(exist_ok=True)
    shutil.copyfile(participants_file, folder / f"{random_lunch_date}-participants")
    with open(folder / f"{random_lunch_date}-config", "w") as f:
        f.write(configuration_str)


def parse_configuration(configuration_file, participants):
    """ Parse a configuration from a generated file. """
    with open(configuration_file, "r") as f:
        configuration = []
        group = []
        for line in f.readlines():
            line = line.lstrip().rstrip()
            # An empty line or a line beginning with "Table" means a new group.
            if line[:len("Table")] == "Table" or line == "":
                if len(group) > 0:
                    configuration.append(group)
                    group = []
                continue
            else:
                group.append(participants.index[participants[name_field] == line][0])
    return configuration


def get_history():
    """ Returns the saved history of random lunches, or an empty one if it doesn't exist. """
    history_file = Path("history")
    if history_file.exists():
        with open("history", "r") as f:
            return History(*json.load(f))
    else:
        return History({}, {}, {})


def save_history(random_lunch_date, history):
    """ Saves the updated history. """
    # Save it in a unique file for backup, if it does not exist yet to avoid registering twice a random lunch.
    assert not Path(f"random_lunches/{random_lunch_date}-history").exists()
    with open(f"random_lunches/{random_lunch_date}-history", "w") as f:
        json.dump(history, f)

    # Also save the history in the global history file for ease of use.
    with open("history", "w") as f:
        json.dump(history, f)


def propose_random_lunch(random_lunch_date, participants_file, group_sizes=None):
    """ Propose a random lunch configuration with the given participants. """
    participants = get_participants(participants_file)
    num_participants = len(participants)
    if group_sizes is None:
        group_sizes = get_group_sizes(num_participants)
    else:
        assert sum(group_size * num_groups for group_size, num_groups in group_sizes.items()) == num_participants
    history = get_history()

    configuration = get_random_minimal_cost_configuration(participants, group_sizes, history.pairs)

    save_configuration(configuration, participants, participants_file, random_lunch_date)


def record_random_lunch(random_lunch_date, cost, payer_team):
    """ Record the past random lunch in the history. """
    participants = get_participants(f"random_lunches/{random_lunch_date}-participants")
    configuration = parse_configuration(f"random_lunches/{random_lunch_date}-config", participants)
    history = get_history()

    update_history(configuration, participants, cost, payer_team, history)

    save_history(random_lunch_date, history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true",
                        help="record a past random lunch (propose a new one otherwise)")
    parser.add_argument("--date", type=str,
                        help="date of the random lunch, in YYYY-MM-DD format (default: next/previous Tuesday)")
    parser.add_argument("--participants", type=str, default="Formulaire sans titre.csv",
                        help="CSV file with the participants data")
    parser.add_argument("--cost", type=float, help="cost of the random lunch (only if record mode)")
    parser.add_argument("--payer-team", type=str, help="team which paid the random lunch (only if record mode)")
    args = parser.parse_args()

    if args.record:
        if args.date is None:  # Previous random lunch day.
            today = datetime.date.today()
            days = (today.weekday() - random_lunch_day) % 7
            args.date = (today - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        record_random_lunch(args.date, args.cost, args.payer_team)
    else:
        if args.date is None:  # Next random lunch day.
            today = datetime.date.today()
            days = (random_lunch_day - today.weekday()) % 7
            args.date = (today + datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        propose_random_lunch(args.date, args.participants, group_sizes=None)
