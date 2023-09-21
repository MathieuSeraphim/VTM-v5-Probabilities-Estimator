import pprint
import numpy as np
from typing import Union, Tuple
from typeguard import typechecked


@typechecked
# Generates the probability distribution of a given dice roll in Vampire: the Masquerade 5th Edition
# This assumes that you are rolling a number of 10-sided dice, and follow the dice roll rules given in the Corebook
# Willpower rerolls will only affect normal dice that didn't produce a success
class RollProbabilityDistributionStatisticalApproximationForVampireTheMasqueradeV5:

    # Generates the dice rolls to compute statistics on the given dice pool
    # total_dice_rolled: the dice pool (number of 10-sided dice to roll)
    # statistical_simulation_size: the number of rolls simulated to estimate the roll's probability distribution
    def __init__(self, total_dice_rolled: int, statistical_simulation_size: int = int(1e6)):
        assert total_dice_rolled > 0
        assert statistical_simulation_size > 0

        # Input constants
        self.__TOTAL_DICE_ROLLED = total_dice_rolled
        self.__STATISTICAL_SIMULATION_SIZE = statistical_simulation_size

        # Generate random dice rolls used in the statistical analysis
        self.__dice_roll_generation()

    # Other public methods

    def get_total_number_of_dice_rolled(self):
        return self.__TOTAL_DICE_ROLLED

    def get_statistical_simulation_size(self):
        return self.__STATISTICAL_SIMULATION_SIZE

    # Re-generates the dice rolls (basically a reset)
    def redo_all_rolls(self):
        self.__dice_roll_generation()

    # A generic function, allowing the user to finetune exactly what they seek as their probability distribution
    # estimation.
    # hunger_value: the character's Hunger level (between 0 and 5)
    # difficulty_value: the difficulty of the roll (minimum of 0)
    # willpower_flag: whether or not the character will spend a point of Willpower to reroll up to 3 normal failures
    # meets_it_beats_it_flag: if set to True, meeting the difficulty exactly is a success; otherwise, it's a failure
    # output_forms: one or more ways that you'd like your data to be formatted
    # The output is a list of objects corresponding to the output_forms - if output_forms is a string, it returns a list
    # of one element
    def get_custom_estimation(self, hunger_value: int, difficulty_value: int, willpower_flag: bool,
                              meets_it_beats_it_flag: bool, output_forms: Union[str, Tuple[str, ...]]):
        # Function constants
        supported_output_forms = self.__SUPPORTED_OUTPUT_FORMS

        # Dealing with the output forms
        if isinstance(output_forms, str):
            output_forms = (output_forms,)
        for output_form in output_forms:
            assert output_form in supported_output_forms

        roll_results_in_vectors_dict = self.__compute_roll_results(hunger_value=hunger_value,
                                                                   difficulty_value=difficulty_value,
                                                                   willpower_flag=willpower_flag,
                                                                   meets_it_beats_it_flag=meets_it_beats_it_flag)

        output = []
        for output_form in  output_forms:
            output.append(self.__apply_form_to_results(roll_results_in_vectors_dict=roll_results_in_vectors_dict,
                                                       output_form=output_form))

        return output

    # Private methods

    # Dice roll generation
    def __dice_roll_generation(self):
        # Aliases
        total_dice_rolled = self.__TOTAL_DICE_ROLLED
        statistical_simulation_size = self.__STATISTICAL_SIMULATION_SIZE
        potential_extra_willpower_dice_rolled = self.__MAX_NUMBER_OF_DICE_REROLLABLE_THROUGH_WILLPOWER
        low = self.__DICE_LOWEST_VALUE
        high = self.__DICE_HIGHEST_VALUE

        # Random rolls
        self.__all_rolls = np.random.randint(low=low, high=high+1, size=(statistical_simulation_size, total_dice_rolled))
        self.__all_potential_willpower_rerolls = \
            np.random.randint(low=low, high=high+1, size=(statistical_simulation_size,
                                                          potential_extra_willpower_dice_rolled))

    def __compute_roll_results(self, hunger_value: int, difficulty_value: int, willpower_flag: bool,
                               meets_it_beats_it_flag: bool):
        # Asserts and aliases
        assert 0 <= hunger_value <= self.__MAX_HUNGER_VALUE
        assert difficulty_value >= 0
        size_of_vectors = self.__STATISTICAL_SIMULATION_SIZE
        total_dice_rolled = self.__TOTAL_DICE_ROLLED
        all_rolls = np.copy(self.__all_rolls)
        success_threshold = self.__DIE_SUCCESS_THRESHOLD
        critical_die_value = self.__DICE_HIGHEST_VALUE
        bestial_die_value = self.__DICE_LOWEST_VALUE
        max_willpower_rerolls = self.__MAX_NUMBER_OF_DICE_REROLLABLE_THROUGH_WILLPOWER
        min_number_of_successes_for_a_successful_roll_key = self.__MIN_NUMBER_OF_SUCCESSES_FOR_A_SUCCESSFUL_ROLL_KEY
        all_successes_vector_key = self.__ALL_SUCCESSES_VECTOR_KEY
        rolls_are_successful_boolean_vector_key = self.__ROLLS_ARE_SUCCESSFUL_BOOLEAN_VECTOR_KEY
        rolls_are_a_critical_success_boolean_vector_key = self.__ROLLS_ARE_A_CRITICAL_SUCCESS_BOOLEAN_VECTOR_KEY
        rolls_are_a_normal_success_boolean_vector_key = self.__ROLLS_ARE_A_NORMAL_SUCCESS_BOOLEAN_VECTOR_KEY
        rolls_are_a_normal_critical_boolean_vector_key = self.__ROLLS_ARE_A_NORMAL_CRITICAL_BOOLEAN_VECTOR_KEY
        rolls_are_a_normal_failure_boolean_vector_key = self.__ROLLS_ARE_A_NORMAL_FAILURE_BOOLEAN_VECTOR_KEY
        rolls_are_a_messy_critical_boolean_vector_key = self.__ROLLS_ARE_A_MESSY_CRITICAL_BOOLEAN_VECTOR_KEY
        rolls_are_a_bestial_failure_boolean_vector_key = self.__ROLLS_ARE_A_BESTIAL_FAILURE_BOOLEAN_VECTOR_KEY

        number_of_hunger_dice = min(hunger_value, total_dice_rolled)
        number_of_normal_dice = total_dice_rolled - number_of_hunger_dice

        # Extra flags
        no_hunger_flag = (hunger_value == 0)
        if no_hunger_flag:
            assert number_of_hunger_dice == 0  # Sanity check

        # Dealing with Willpower rerolls
        if willpower_flag:
            willpower_rerolls = np.copy(self.__all_potential_willpower_rerolls)
            all_normal_dice_rolls = all_rolls[:, :number_of_normal_dice]

            # Creating a mask to detect rolls with at least one, two or three normal dice failures
            # (and thus able to reroll at least 1, 2 or 3 dice, respectively)
            all_normal_dice_failures = np.sum(all_normal_dice_rolls < success_threshold, axis=-1)
            at_least_n_failures_boolean_vectors_list = []
            for i in range(1, max_willpower_rerolls+1):
                at_least_n_failures_boolean_vectors_list.append(all_normal_dice_failures >= i)
            willpower_mask = np.stack(at_least_n_failures_boolean_vectors_list, axis=1)

            # Apply the mask and combine with the normal rolls
            assert willpower_mask.shape == willpower_rerolls.shape
            willpower_rerolls = willpower_rerolls * willpower_mask
            all_rolls = np.concatenate((willpower_rerolls, all_rolls), axis=1)
            number_of_normal_dice += max_willpower_rerolls

        # Accounting for criticals (pairs of 10s that count as 4 successes)
        all_successes_not_counting_criticals_vector = np.sum(all_rolls >= success_threshold, axis=-1)
        all_crits_vector = np.sum(all_rolls == critical_die_value, axis=-1)
        nb_of_critical_pairs_vector = np.floor(all_crits_vector / 2)
        additional_successes_from_criticals_vector = nb_of_critical_pairs_vector * 2
        all_successes_vector = all_successes_not_counting_criticals_vector + additional_successes_from_criticals_vector

        if meets_it_beats_it_flag:
            min_number_of_successes_for_a_successful_roll = difficulty_value
        else:
            min_number_of_successes_for_a_successful_roll = difficulty_value + 1
        rolls_are_successful_boolean_vector = all_successes_vector >= min_number_of_successes_for_a_successful_roll

        rolls_contain_critical_pairs_boolean_vector = (all_crits_vector >= 2)
        rolls_are_a_critical_success_boolean_vector = rolls_are_successful_boolean_vector\
                                                    & rolls_contain_critical_pairs_boolean_vector

        # No messy crits, no bestial dice
        if no_hunger_flag:
            rolls_contain_crits_on_hunger_dice_boolean_vector = np.zeros(size_of_vectors, dtype=bool)
            rolls_contain_bestial_hunger_dice_boolean_vector = np.zeros(size_of_vectors, dtype=bool)

        else:
            all_hunger_dice_rolls = all_rolls[:, number_of_normal_dice:]
            assert all_hunger_dice_rolls.shape[1] == number_of_hunger_dice  # Sanity check
            rolls_contain_crits_on_hunger_dice_boolean_vector = (all_hunger_dice_rolls == critical_die_value)\
                .any(axis=-1)
            rolls_contain_bestial_hunger_dice_boolean_vector = (all_hunger_dice_rolls == bestial_die_value).any(axis=-1)

        # Mutually exclusive boolean vectors (all categories)
        rolls_are_a_normal_success_boolean_vector = rolls_are_successful_boolean_vector & np.logical_not(
                                                    rolls_are_a_critical_success_boolean_vector)
        rolls_are_a_normal_critical_boolean_vector = rolls_are_a_critical_success_boolean_vector & np.logical_not(
                                                     rolls_contain_crits_on_hunger_dice_boolean_vector)
        rolls_are_a_normal_failure_boolean_vector = np.logical_not(rolls_are_successful_boolean_vector)\
                                                  & np.logical_not(rolls_contain_bestial_hunger_dice_boolean_vector)
        rolls_are_a_messy_critical_boolean_vector = rolls_are_a_critical_success_boolean_vector\
                                                  & rolls_contain_crits_on_hunger_dice_boolean_vector
        rolls_are_a_bestial_failure_boolean_vector = np.logical_not(rolls_are_successful_boolean_vector)\
                                                   & rolls_contain_bestial_hunger_dice_boolean_vector

        # The sanity-checkest of sanity checks - checking if the boolean values are indeed exclusive
        sanity_check_vector = rolls_are_a_normal_success_boolean_vector.astype(int)\
                            + rolls_are_a_normal_critical_boolean_vector.astype(int)\
                            + rolls_are_a_normal_failure_boolean_vector.astype(int)\
                            + rolls_are_a_messy_critical_boolean_vector.astype(int)\
                            + rolls_are_a_bestial_failure_boolean_vector.astype(int)
        assert (sanity_check_vector == 1).all()
        assert len(sanity_check_vector.shape) == 1
        assert sanity_check_vector.shape[0] == size_of_vectors

        roll_results_in_vectors_dict = {
            min_number_of_successes_for_a_successful_roll_key: min_number_of_successes_for_a_successful_roll,
            all_successes_vector_key: all_successes_vector,
            rolls_are_successful_boolean_vector_key: rolls_are_successful_boolean_vector,
            rolls_are_a_critical_success_boolean_vector_key: rolls_are_a_critical_success_boolean_vector,
            rolls_are_a_normal_success_boolean_vector_key: rolls_are_a_normal_success_boolean_vector,
            rolls_are_a_normal_critical_boolean_vector_key: rolls_are_a_normal_critical_boolean_vector,
            rolls_are_a_normal_failure_boolean_vector_key: rolls_are_a_normal_failure_boolean_vector,
        }

        if not no_hunger_flag:
            roll_results_in_vectors_dict[rolls_are_a_messy_critical_boolean_vector_key]\
                = rolls_are_a_messy_critical_boolean_vector
            roll_results_in_vectors_dict[rolls_are_a_bestial_failure_boolean_vector_key]\
                = rolls_are_a_bestial_failure_boolean_vector

        return roll_results_in_vectors_dict

    def __apply_form_to_results(self, roll_results_in_vectors_dict, output_form):
        assert output_form in self.__SUPPORTED_OUTPUT_FORMS

        if output_form == "raw_results_dict":
            return roll_results_in_vectors_dict

        if output_form == "basic_statistics_dict":
            return self.__compute_basic_statistics_from_results(
                roll_results_in_vectors_dict=roll_results_in_vectors_dict)

        if output_form == "probability_distribution_dict":
            return self.__compute_probability_distribution_from_results(
                roll_results_in_vectors_dict=roll_results_in_vectors_dict)

        raise NotImplementedError

    def __compute_basic_statistics_from_results(self, roll_results_in_vectors_dict):
        # Aliases and constants
        number_of_rolls = self.__STATISTICAL_SIMULATION_SIZE
        successes_boolean_vector_key = self.__ROLLS_ARE_SUCCESSFUL_BOOLEAN_VECTOR_KEY
        critical_successes_boolean_vector_key = self.__ROLLS_ARE_A_CRITICAL_SUCCESS_BOOLEAN_VECTOR_KEY
        messy_criticals_boolean_vector_key = self.__ROLLS_ARE_A_MESSY_CRITICAL_BOOLEAN_VECTOR_KEY
        bestial_failures_boolean_vector_key = self.__ROLLS_ARE_A_BESTIAL_FAILURE_BOOLEAN_VECTOR_KEY
        results_dict_keys = roll_results_in_vectors_dict

        # Calculations
        successes_boolean_vector = roll_results_in_vectors_dict[successes_boolean_vector_key]
        critical_successes_boolean_vector = roll_results_in_vectors_dict[critical_successes_boolean_vector_key]
        basic_statistics_dict = {
            "Proportion of roll Successes": np.sum(successes_boolean_vector) / number_of_rolls,
            "Proportion of Critical Successes": np.sum(critical_successes_boolean_vector) / number_of_rolls,
        }

        # Adding Hunger complication statistics, if any
        if messy_criticals_boolean_vector_key in results_dict_keys\
                and bestial_failures_boolean_vector_key in results_dict_keys:
            messy_criticals_boolean_vector = roll_results_in_vectors_dict[messy_criticals_boolean_vector_key]
            bestial_failures_boolean_vector = roll_results_in_vectors_dict[bestial_failures_boolean_vector_key]
            basic_statistics_dict["Proportion of Messy Criticals"]\
                = np.sum(messy_criticals_boolean_vector) / number_of_rolls
            basic_statistics_dict["Proportion of Bestial Failures"] \
                = np.sum(bestial_failures_boolean_vector) / number_of_rolls

        return basic_statistics_dict

    def __compute_probability_distribution_from_results(self, roll_results_in_vectors_dict):
        # Aliases and constants
        number_of_dice_rolled = self.__TOTAL_DICE_ROLLED
        number_of_rolls = self.__STATISTICAL_SIMULATION_SIZE
        roll_difficulty_threshold_key = self.__MIN_NUMBER_OF_SUCCESSES_FOR_A_SUCCESSFUL_ROLL_KEY
        roll_difficulty_threshold = roll_results_in_vectors_dict[roll_difficulty_threshold_key]

        # Should be equal to number_of_rolls by the end of the function
        sanity_check_counter = 0

        # Defining the range of possible outputs, knowing that every pair of 10s counts as 4 successes
        if number_of_dice_rolled % 2 == 0:  # Even number of dice rolled
            max_possible_number_of_successes = 2 * number_of_dice_rolled
        else:  # Odd number of dice rolled
            max_possible_number_of_successes = 2 * (number_of_dice_rolled - 1) + 1
        # All integers in [0, max_possible_number_of_successes]
        possible_numbers_of_successes_range = tuple(range(max_possible_number_of_successes+1))

        # TODO Continue!
        raise NotImplementedError

    # Constants

    # Normal constants
    __DICE_LOWEST_VALUE = 1
    __DICE_HIGHEST_VALUE = 10
    __DIE_SUCCESS_THRESHOLD = 6
    __MAX_HUNGER_VALUE = 5
    __MAX_NUMBER_OF_DICE_REROLLABLE_THROUGH_WILLPOWER = 3
    __SUPPORTED_OUTPUT_FORMS = ("raw_results_dict", "basic_statistics_dict", "probability_distribution_dict",)

    # Dictionary keys
    __MIN_NUMBER_OF_SUCCESSES_FOR_A_SUCCESSFUL_ROLL_KEY = "Minimum number of successes for the roll to be a success"
    __ALL_SUCCESSES_VECTOR_KEY = "Successes per roll"
    __ROLLS_ARE_SUCCESSFUL_BOOLEAN_VECTOR_KEY = "Successful rolls"
    __ROLLS_ARE_A_CRITICAL_SUCCESS_BOOLEAN_VECTOR_KEY = "Critically successful rolls"
    __ROLLS_ARE_A_NORMAL_SUCCESS_BOOLEAN_VECTOR_KEY = "Successful rolls without critical pairs"
    __ROLLS_ARE_A_NORMAL_CRITICAL_BOOLEAN_VECTOR_KEY = "Critically successful and non-messy rolls"
    __ROLLS_ARE_A_NORMAL_FAILURE_BOOLEAN_VECTOR_KEY = "Non-bestially failed rolls"
    __ROLLS_ARE_A_MESSY_CRITICAL_BOOLEAN_VECTOR_KEY = "Messy critically successful rolls"
    __ROLLS_ARE_A_BESTIAL_FAILURE_BOOLEAN_VECTOR_KEY = "Bestially failed rolls"


# Using inheritance to create an alias for the class
class ProbabilitiesForV5(RollProbabilityDistributionStatisticalApproximationForVampireTheMasqueradeV5):
    pass


if __name__ == "__main__":
    my_pool_size = 17
    hunger_levels = list(range(6))
    difficulty = 6
    spending_willpower = True
    is_active_player = True
    number_of_test_rolls = 10000000
    desired_output_forms = ("basic_statistics_dict",)# "probability_distribution_dict")  # WIP

    my_roll_results_estimator = ProbabilitiesForV5(total_dice_rolled=my_pool_size,
                                                   statistical_simulation_size=number_of_test_rolls)

    print("Custom estimator test\n")
    for hunger_level in hunger_levels:
        print("  For Hunger %d:" % hunger_level)

        my_output = my_roll_results_estimator.get_custom_estimation(hunger_value=hunger_level,
                                                                    difficulty_value=difficulty,
                                                                    willpower_flag=spending_willpower,
                                                                    meets_it_beats_it_flag=is_active_player,
                                                                    output_forms=desired_output_forms)

        for desired_output_form_index in range(len(desired_output_forms)):
            print("    Output of form %s:" % desired_output_forms[desired_output_form_index])
            pprint.pprint((my_output[desired_output_form_index]), indent=6)
            print()



