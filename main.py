""" Random lunch script.

Participants are identified by a string (their email address). They are associated to a team.
"""

import pickle
import random
import datetime
from pathlib import Path


# Problem definition

ideal_group_size = 4  # Try to make groups of this size.


"""
We measure the cost of a configuration as the sum over all pairs of participants in a same group,
of the squared number of times they have already been in a same group.
People in the same team are considered to have spent 1000 lunches together.
A history is then a dictionary from pairs of participants to the number of times they have had lunch together. 
"""


def get_cost(configuration, history):
    """ Measure the cost of a configuration according to past history of random lunches.
    :param configuration: list of lists of email addresses.
    :param history: dictionary of sorted email address pairs to number of past lunches.
    :return: the cost of the configuration (an integer), the smaller the better.
    """
    cost = 0
    for pair in enumerate_pairs(configuration):
        num_times = history.get(pair, 0)
        if get_team(pair[0]) == get_team(pair[1]):
            num_times += 1000
        cost += num_times ** 2
    return cost


def update_history(configuration, history):
    """ Updates in-place the history to account for the chosen configuration. """
    for pair in enumerate_pairs(configuration):
        # Record the pair in the history.
        history[pair] = history.get(pair, 0) + 1


def enumerate_pairs(configuration):
    """ Returns sorted pairs of participants which are in a same group. """
    for group in configuration:
        for pair in enumerate_groups(num_participants=len(group), group_size=2):
            # Translate the pair from numbers to email addresses for the history.
            # We are guaranteed that it is sorted, as both group and pair and increasing lists.
            yield tuple([group[participant] for participant in pair])


empty_history = {}


# Enumeration of configurations.

def get_group_sizes(num_participants):
    """ Break the number of participants into group sizes.
    We try to make groups that are of size ideal_group_size or ideal_group_size - 1.
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

    # Otherwise, we try to make groups of size ideal_group_size - 1.
    # This is possible if there are enough full groups: we move some participants from them to this last group.
    elif num_full_groups >= (num_participants_to_move := ideal_group_size - 1 - rest):
        groups = {ideal_group_size: num_full_groups - num_participants_to_move,
                  ideal_group_size - 1: num_participants_to_move + 1}

    # If this is not possible, we make larger groups.
    # We then move participants from the rest group to the full groups.
    else:
        additional_participants_per_group = rest // num_full_groups  # Should be zero most of the time.
        rest_rest = rest % num_full_groups
        groups = {ideal_group_size + additional_participants_per_group: num_full_groups - rest_rest,
                  ideal_group_size + additional_participants_per_group + 1: rest_rest}

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


def test_enumerate_configs():
    """ Small function to test the enumeration of configurations. """
    num_participants = 7
    group_sizes = get_group_sizes(num_participants)
    print(num_participants, group_sizes)
    configurations = list(enumerate_configurations(num_participants, group_sizes))
    for configuration in configurations:
        assert sorted(sum(configuration, start=[])) == list(range(num_participants))
        print(configuration)


def get_random_minimal_cost_configuration(participants, group_sizes, history):
    """ Returns a random configuration among those of minimal cost. """
    num_participants = len(participants)

    # Compute all configuration of with minimal cost.
    best_cost = float("inf")
    best_configurations = []
    for configuration in enumerate_configurations(num_participants, group_sizes):
        # Translate participants from numbers to email addresses.
        configuration = [[participants[participant] for participant in group] for group in configuration]
        cost = get_cost(configuration, history)

        if cost < best_cost:
            best_cost = cost
            best_configurations = [configuration]
        elif cost == best_cost:
            best_configurations.append(configuration)

    # Return a random configuration among all best ones.
    return best_configurations[random.randrange(0, len(best_configurations))]


def get_participants():
    """ Returns the list of email addresses of today's participants. """
    raise NotImplementedError


def get_team(participant):
    """ Returns the team of the participant. """
    raise NotImplementedError


def send_emails(configuration):
    """ Send emails to each group in the input configuration. """
    raise NotImplementedError


def get_history():
    """ Returns the saved history of random lunches, or an empty one if it doesn't exist. """
    history_file = Path("history")
    if history_file.exists():
        with open("history", "rb") as f:
            return pickle.load(f)
    else:
        return empty_history


def save_configuration_and_history(configuration, history):
    """ Saves today's chosen configuration (just in case) and the updated history. """
    with open("history", "wb") as f:
        print("history", history)
        pickle.dump(history, f)

    folder = Path("random_lunches")
    folder.mkdir(exist_ok=True)
    today_file_name = datetime.date.today().strftime("%Y-%m-%d")
    with open(folder / today_file_name, "wb") as f:
        pickle.dump(configuration, f)


def random_lunch():
    """ Main function which takes care of everything. """
    participants = list(sorted(get_participants()))
    num_participants = len(participants)
    group_sizes = get_group_sizes(num_participants)
    history = get_history()

    configuration = get_random_minimal_cost_configuration(participants, group_sizes, history)

    update_history(configuration, history)
    save_configuration_and_history(configuration, history)
    send_emails(configuration)


if __name__ == "__main__":
    random_lunch()
