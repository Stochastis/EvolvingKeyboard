#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Evolve a better keyboard.
This assignment is mostly open-ended,
with a couple restrictions:

# DO NOT MODIFY >>>>
Do not edit the sections between these marks below.
# <<<< DO NOT MODIFY
"""

# %%
import random
from typing import TypedDict
import math
import json

# Run the following command as written for the proper version of 'playsound':
# pip install playsound==1.2.2
from playsound import playsound

# Constants (Parameters)
STARTINGMUTATERATE = 0.0001
MINSTARTINGFITNESS = 32.2
GENS = 50000

# Region for the "DO NOT MODIFY" section
# region
# DO NOT MODIFY >>>>
# First, what should our representation look like?
# Is there any modularity in adjacency?
# What mechanisms capitalize on such modular patterns?
# ./corpus/2_count.py specificies this same structure
# Positions    01234   56789   01234
LEFT_DVORAK = "',.PY" "AOEUI" ";QJKX"
LEFT_QWERTY = "QWERT" "ASDFG" "ZXCVB"
LEFT_COLEMK = "QWFPG" "ARSTD" "ZXCVB"
LEFT_WORKMN = "QDRWB" "ASHTG" "ZXMCV"

LEFT_DISTAN = "11111" "00001" "11111"
LEFT_ERGONO = "00001" "00001" "11212"
LEFT_EDGE_B = "01234" "01234" "01234"

# Positions     56   7890123   456789   01234
RIGHT_DVORAK = "[]" "FGCRL/=" "DHTNS-" "BMWVZ"
RIGHT_QWERTY = "-=" "YUIOP[]" "HJKL;'" "NM,./"
RIGHT_COLEMK = "-=" "JLUY;[]" "HNEIO'" "KM,./"
RIGHT_WOKRMN = "-=" "JFUP;[]" "YNEOI'" "KL,./"

RIGHT_DISTAN = "23" "1111112" "100000" "11111"
RIGHT_ERGONO = "22" "2000023" "100001" "10111"
RIGHT_EDGE_B = "10" "6543210" "543210" "43210"

DVORAK = LEFT_DVORAK + RIGHT_DVORAK
QWERTY = LEFT_QWERTY + RIGHT_QWERTY
COLEMAK = LEFT_COLEMK + RIGHT_COLEMK
WORKMAN = LEFT_WORKMN + RIGHT_WOKRMN

DISTANCE = LEFT_DISTAN + RIGHT_DISTAN
ERGONOMICS = LEFT_ERGONO + RIGHT_ERGONO
PREFER_EDGES = LEFT_EDGE_B + RIGHT_EDGE_B

# Real data on w.p.m. for each letter, normalized.
# Higher values is better (higher w.p.m.)
with open(file="typing_data/manual-typing-data_qwerty.json", mode="r") as f:
    data_qwerty = json.load(fp=f)
with open(file="typing_data/manual-typing-data_dvorak.json", mode="r") as f:
    data_dvorak = json.load(fp=f)
data_values = list(data_qwerty.values()) + list(data_dvorak.values())
mean_value = sum(data_values) / len(data_values)
data_combine = []
for dv, qw in zip(DVORAK, QWERTY):
    if dv in data_dvorak.keys() and qw in data_qwerty.keys():
        data_combine.append((data_dvorak[dv] + data_qwerty[qw]) / 2)
    if dv in data_dvorak.keys() and qw not in data_qwerty.keys():
        data_combine.append(data_dvorak[dv])
    if dv not in data_dvorak.keys() and qw in data_qwerty.keys():
        data_combine.append(data_qwerty[qw])
    else:
        # Fill missing data with the mean
        data_combine.append(mean_value)


class Individual(TypedDict):
    genome: str
    fitness: int


Population = list[Individual]


def print_keyboard(individual: Individual) -> None:
    layout = individual["genome"]
    fitness = individual["fitness"]
    """Prints the keyboard in a nice way"""
    print("______________  ________________")
    print(" ` 1 2 3 4 5 6  7 8 9 0 " + " ".join(layout[15:17]) + " Back")
    print("Tab " + " ".join(layout[0:5]) + "  " + " ".join(layout[17:24]) + " \\")
    print("Caps " + " ".join(layout[5:10]) + "  " + " ".join(layout[24:30]) + " Enter")
    print(
        "Shift " + " ".join(layout[10:15]) + "  " + " ".join(layout[30:35]) + " Shift"
    )
    print(f"\nAbove keyboard has fitness of: {fitness}")


# <<<< DO NOT MODIFY
# endregion


def initIndividual(genome: str, fitness: int) -> Individual:
    """
    Purpose:        Create one individual
    Parameters:     genome as string, fitness as integer (higher better)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a dict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    return Individual(genome=genome, fitness=fitness)


def initPop(popSize: int) -> Population:
    """
    Purpose:        Create a randomized population to evolve
    Parameters:     Population size as int
    User Input:     no
    Prints:         no
    Returns:        a population, as a list of Individuals
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    result = []
    keys = DVORAK
    for _ in range(popSize):
        keys = "".join(random.sample(keys, len(keys)))
        result.append(Individual(genome=keys, fitness=0))
    return result


def mapFunction(index: int, c1NewSegment: list[str], c2NewSegment: list[str]) -> str:
    """
    Follows the trail in a pair of genome sections for mapping/legalization purposes
    """
    c2Key = c2NewSegment[index]
    if c2Key not in c1NewSegment:
        return c2Key
    index = c1NewSegment.index(c2Key)
    return mapFunction(index, c1NewSegment, c2NewSegment)


def recombinePair(
    parent1: Individual, parent2: Individual, GENOMESIZE: int
) -> Population:
    """
    Purpose:        Recombine two parents to produce two children
    Parameters:     Two parents as Individuals
    User Input:     no
    Prints:         no
    Returns:        A population of size 2, the children
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    p1G = list(parent1["genome"])
    p2G = list(parent2["genome"])

    # Select crossover range at random
    begin = random.randrange(0, GENOMESIZE - 1)
    end = random.randrange(begin + 1, GENOMESIZE)

    # Create offspring by exchanging genetic information between parents
    c1NewSegment = p2G[begin : end + 1]
    c2NewSegment = p1G[begin : end + 1]
    c1G = p1G[:begin] + c1NewSegment + p1G[end + 1 :]
    c2G = p2G[:begin] + c2NewSegment + p2G[end + 1 :]

    # Determine mapping relationship to legalize offspring
    mapping = {}
    for i in range(len(c1NewSegment)):
        currKey1 = c1NewSegment[i]
        if currKey1 not in c2NewSegment:
            mapping[currKey1] = mapFunction(i, c1NewSegment, c2NewSegment)
            mapping[mapping[currKey1]] = currKey1

    # Legalize children with the mapping relationship
    for i in range(GENOMESIZE):
        if i < begin or i > end:
            if c1G[i] in mapping.keys():
                c1G[i] = mapping[c1G[i]]
            if c2G[i] in mapping.keys():
                c2G[i] = mapping[c2G[i]]

    if len(c1G) == len(set(c1G)) and len(c2G) == len(set(c2G)):
        child1 = Individual(genome="".join(c1G), fitness=parent1["fitness"])
        child2 = Individual(genome="".join(c2G), fitness=parent2["fitness"])
        return Population([child1, child2])
    else:
        print("DUPLICATE KEYS DETECTED!!!")
        print(c1G)
        print(c2G)
        exit()


def recombineGroup(
    parents: Population, recombine_rate: float, GENOMESIZE: int
) -> Population:
    """
    Purpose:        Recombines a whole group, returns the new population
                    Pair parents 1-2, 2-3, 3-4, etc..
                    Recombine at rate, else clone the parents.
    Parameters:     parents and recombine rate
    User Input:     no
    Prints:         no
    Returns:        New population of children
    Modifies:       Nothing
    Calls:          ?
    """
    newPop = Population()
    for i in range(0, len(parents), 2):
        p1 = parents[i]
        p2 = parents[i + 1]
        if not random.random() < recombine_rate:
            newPop.extend([p1, p2])
        else:
            newPop.extend(recombinePair(p1, p2, GENOMESIZE))

    return newPop


def mutateIndividual(
    parent: Individual, mutate_rate: float, GENOMESIZE: int
) -> Individual:
    """
    Purpose:        Mutate one individual
    Parameters:     One parent as an Individual, mutation rate as a float (0-1)
    User Input:     no
    Prints:         no
    Returns:        One Individual, as a TypedDict[str, int]
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    result = Individual(genome=parent["genome"], fitness=parent["fitness"])
    for i in range(GENOMESIZE):
        if random.random() < mutate_rate:
            j = random.randrange(0, GENOMESIZE)
            l = list(result["genome"])
            l[i], l[j] = l[j], l[i]
            result["genome"] = "".join(l)

    return result


def mutateGroup(
    children: Population, mutate_rate: float, GENOMESIZE: int
) -> Population:
    """
    Purpose:        Mutates a whole Population, returns the mutated group
    Parameters:     Population, mutation rate as float (0-1)
    User Input:     no
    Prints:         no
    Returns:        Mutated population
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    newPop = Population()
    for child in children:
        newPop.append(mutateIndividual(child, mutate_rate, GENOMESIZE=GENOMESIZE))
    return newPop


# Region for the "DO NOT MODIFY" section
# region
# DO NOT MODIFY >>>>


def evaluate_individual(individual: Individual) -> None:
    """
    Purpose:        Computes and modifies the fitness for one individual
                    Assumes and relies on the logc of ./corpus/2_counts.py
    Parameters:     One Individual
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The individual (mutable object)
    Calls:          Basic python only
    Example doctest:
    """
    layout = individual["genome"]

    # Basic return to home row, with no extra cost for repeats.
    fitness = 0
    for key in layout:
        fitness += count_dict[key] * int(DISTANCE[layout.find(key)])

    # Vowels on the left, Consosants on the right
    for pos, key in enumerate(layout):
        if key in "AEIOUY" and pos > 14:
            fitness += 1

    # Top-down guess at ideal ergonomics
    for key in layout:
        fitness += count_dict[key] * int(ERGONOMICS[layout.find(key)])

    # [] {} () <> should be adjacent.
    # () ar fixed by design choice (number line).
    # [] and {} are on same keys.
    # Perhaps ideally, <> and () should be on same keys too...
    right_edges = [4, 9, 14, 16, 23, 29, 34]
    for pos, key in enumerate(layout):
        # order of (x or y) protects index on far right:
        if key == "[" and (pos in right_edges or "]" != layout[pos + 1]):
            fitness += 1
        if key == "," and (pos in right_edges or "." != layout[pos + 1]):
            fitness += 1

    # Symbols should be toward edges.
    for pos, key in enumerate(layout):
        if key in "-[],.';/=":
            fitness += int(PREFER_EDGES[pos])

    # Keybr.com querty-dvorak average data as estimate of real hand
    for pos, key in enumerate(layout):
        fitness += count_dict[key] / data_combine[pos]

    # Shortcut characters (skip this one).
    # On right hand for keyboarders (left ctrl is usually used)
    # On left hand for mousers (for one-handed shortcuts).
    pass

    individual["fitness"] = fitness


# <<<< DO NOT MODIFY
# endregion


def evalGroup(individuals: Population) -> None:
    """
    Purpose:        Computes and modifies the fitness for population
    Parameters:     Objective string, Population
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The Individuals, all mutable objects
    Calls:          ?
    Example doctest:
    """
    for individual in individuals:
        evaluate_individual(individual)


def rankGroup(individuals: Population) -> None:
    """
    Purpose:        Create one individual
    Parameters:     Population of Individuals
    User Input:     no
    Prints:         no
    Returns:        None
    Modifies:       The population's order (a mutable object)
    Calls:          ?
    Example doctest:
    """
    individuals.sort(key=lambda d: d["fitness"])


def weightedSampleWithoutReplacement(
    population: Population, weights: list[float], k: int = 1
) -> Population:
    """
    Return a sub-list of population of size k without replacement with weights.
    """
    weights = list(weights)
    positions = range(len(population))
    indices: list[int] = []
    while True:
        needed = k - len(indices)
        if not needed:
            break
        for i in random.choices(positions, weights, k=needed):
            if weights[i]:
                weights[i] = 0.0
                indices.append(i)
    return [population[i] for i in indices]


def parentSelect(individuals: Population, number: int) -> Population:
    """
    Purpose:        Choose parents in direct probability to their fitness
    Parameters:     Population, the number of individuals to pick.
    User Input:     no
    Prints:         no
    Returns:        Sub-population
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    if number > len(individuals):
        print("Cannot make a sub-population larger than the original population.")
        exit()
    weights: list[float] = []
    for i in range(len(individuals) - 1, -1, -1):
        # Cubing fitnesses for the weights since we're dealing with small changes in fitness
        # Appending weights in reverse order since a smaller fitness is better
        weights.append(individuals[i]["fitness"] ** 3)

    return weightedSampleWithoutReplacement(individuals, weights, number)


def survivorSelect(individuals: Population, pop_size: int) -> Population:
    """
    Purpose:        Picks who gets to live!
    Parameters:     Population, and population size to return.
    User Input:     no
    Prints:         no
    Returns:        Population, of pop_size
    Modifies:       Nothing
    Calls:          ?
    Example doctest:
    """
    rankGroup(individuals)
    return individuals[:pop_size]


def getInitialPopulation(popSize: int) -> Population:
    """
    Create randomized populations until finding one with a fitness of 30 or less before continuing.
    The idea is to cast a very wide net, land on a 'hill' not discovered yet (30 fitness or less without evolving),
    and continue from there in the hopes of having found a different, better local optima.
    """

    attempt = 1
    foundBestAttempt = 1
    bestFitness = 100
    bigPopSize = popSize * 10

    while True:
        startingPop = initPop(bigPopSize)
        evalGroup(startingPop)
        rankGroup(startingPop)

        if startingPop[0]["fitness"] < bestFitness:
            bestFitness = startingPop[0]["fitness"]
            foundBestAttempt = attempt
            playsound("sounds/tinyImprovement.mp3")
            print("New Best Starting Fitness Found on Attempt {}:".format(attempt))
            print(str(bestFitness) + "\n")

        if startingPop[0]["fitness"] < MINSTARTINGFITNESS:
            playsound("sounds/hitMinFitness.mp3")
            print("FOUND THE GOOD STUFF!!!")
            print("Moving on to standard evolution.\n")
            break

        if attempt % 100 == 0:
            print("Current Attempt: {}".format(attempt))
            print(
                "Best Starting Fitness So Far: Fitness: {} | Attempt: {}\n".format(
                    bestFitness, foundBestAttempt
                )
            )
            playsound("sounds/1000Generations.mp3")

        attempt += 1

    startingPop = survivorSelect(startingPop, popSize)
    return startingPop


def getBestEverKeyboard() -> Individual:
    # Record past best ever keyboard
    bestEverKeyboard = Individual(genome="", fitness=100)

    with open(file="best_ever.txt", mode="r") as f:
        bestEverKeyboard["genome"] = f.readlines()[0]

    evaluate_individual(bestEverKeyboard)
    return bestEverKeyboard


def evolve(exampleGenome: str, popSize: int = 500) -> Population:
    """
    Purpose:        A whole EC run, main driver
    Parameters:     The number of individuals in a population
    User Input:     No
    Prints:         Updates every time fitness switches.
    Returns:        Population
    Modifies:       Various data structures
    Calls:          Basic python, all your functions
    """

    # Initialization/Parameter Tuning
    # region
    GENOMESIZE = len(exampleGenome)

    currPop = getInitialPopulation(popSize)
    bestFitnessThisRun = currPop[0]["fitness"]
    mutateRate = STARTINGMUTATERATE
    sprinkle = initPop(1)[0]
    seedInvividual = Individual(genome="", fitness=0)
    bestEverKeyboard = getBestEverKeyboard()
    # endregion

    # User-Inputed 'Seed' Genome
    # region
    if seedInvividual["genome"] == "":
        pass
    elif sorted(exampleGenome) != sorted(seedInvividual["genome"]):
        print("INVALID SEED STRING!!!")
        exit()
    else:
        currPop[-1] = seedInvividual
        evalGroup(currPop)
    # endregion

    # The main evolution engine
    # region
    for gen in range(GENS):

        # Sprinkle
        currPop[-1] = sprinkle

        # Every 1000 generations, give an update to the user and change sprinkle
        if gen % 1000 == 0:
            playsound("sounds/1000Generations.mp3")
            print("Generation {} Mutation rate:{}".format(gen, mutateRate))
            sprinkle = initPop(1)[0]

        # Main parent -> offspring -> mutation -> survivor cycle
        # region
        parents = parentSelect(currPop, popSize // 2)
        childPop = recombineGroup(parents, 0.75, GENOMESIZE)
        parents = parentSelect(currPop, popSize // 2)
        childPop.extend(recombineGroup(parents, 1, GENOMESIZE))
        currPop.extend(mutateGroup(childPop, mutateRate, GENOMESIZE))
        evalGroup(currPop)
        rankGroup(currPop)
        currPop = survivorSelect(currPop, len(currPop) // 2)
        # endregion

        # Check for better fitness, record new best if necessary, and adjust mutation rate
        # region
        if currPop[0]["fitness"] < bestFitnessThisRun:
            bestFitnessThisRun = currPop[0]["fitness"]
            mutateRate = STARTINGMUTATERATE
            playsound("sounds/tinyImprovement.mp3")
            print("New Best Fitness Found on Generation {}:".format(gen))
            print(bestFitnessThisRun)

            if bestFitnessThisRun == bestEverKeyboard["fitness"]:
                playsound("sounds/hitRecordFitness.mp3")

            if bestFitnessThisRun < bestEverKeyboard["fitness"]:
                # Write the data of the new best keyboard in the file
                with open(file="best_ever.txt", mode="w") as f:
                    f.write(currPop[0]["genome"] + "\n")
                    f.write(str(currPop[0]["fitness"]))

                # Copy new bestEverKeyboard for future comparisons in this run
                bestEverKeyboard = currPop[0]

                playsound("sounds/brokeRecordFitness.mp3")
                print("BROKEN RECORD!!!")
        else:
            mutateRate = min(0.9999, max(0.0001, mutateRate + 0.0000025))
        # endregion
    # endregion

    return currPop


seed = False

# Region for the "DO NOT MODIFY" section
# region
# DO NOT MODIFY >>>>
if __name__ == "__main__":
    divider = "===================================================="
    # Execute doctests to protect main:
    # import doctest

    # doctest.testmod()
    # doctest.testmod(verbose=True)

    if seed:
        random.seed(42)

    with open("corpus/counts.json") as fhand:
        count_dict = json.load(fhand)

    # print("Counts of characters in big corpus, ordered by freqency:")
    # ordered = sorted(count_dict, key=count_dict.__getitem__, reverse=True)
    # for key in ordered:
    #     print(key, count_dict[key])

    print(divider)
    print(
        f"Number of possible permutations of standard keyboard: {math.factorial(len(DVORAK)):,e}"
    )
    print("That's a huge space to search through")
    print("The messy landscape is a difficult to optimize multi-modal space")
    print("Lower fitness is better.")

    print(divider)
    print("\nThis is the Dvorak keyboard:")
    dvorak = Individual(genome=DVORAK, fitness=0)
    evaluate_individual(dvorak)
    print_keyboard(dvorak)

    print(divider)
    print("\nThis is the Workman keyboard:")
    workman = Individual(genome=WORKMAN, fitness=0)
    evaluate_individual(workman)
    print_keyboard(workman)

    print(divider)
    print("\nThis is the Colemak keyboard:")
    colemak = Individual(genome=COLEMAK, fitness=0)
    evaluate_individual(colemak)
    print_keyboard(colemak)

    print(divider)
    print("\nThis is the QWERTY keyboard:")
    qwerty = Individual(genome=QWERTY, fitness=0)
    evaluate_individual(qwerty)
    print_keyboard(qwerty)

    print(divider)
    print("\nThis is a random layout:")
    badarr = list(DVORAK)
    random.shuffle(badarr)
    badstr = "".join(badarr)
    badkey = Individual(genome=badstr, fitness=0)
    evaluate_individual(badkey)
    print_keyboard(badkey)

    print(divider)
    input("Press any key to start")
    population = evolve(exampleGenome=DVORAK)

    print("Here is the best layout:")
    print_keyboard(population[0])

    grade = 0
    if qwerty["fitness"] < population[0]["fitness"]:
        grade = 0
    if colemak["fitness"] < population[0]["fitness"]:
        grade = 50
    if workman["fitness"] < population[0]["fitness"]:
        grade = 60
    elif dvorak["fitness"] < population[0]["fitness"]:
        grade = 70
    else:
        grade = 80

    with open(file="results.txt", mode="w") as f:
        f.write(str(grade))

    with open(file="best_ever.txt", mode="r") as f:
        past_record = f.readlines()[1]
    if population[0]["fitness"] < float(past_record):
        with open(file="best_ever.txt", mode="w") as f:
            f.write(population[0]["genome"] + "\n")
            f.write(str(population[0]["fitness"]))
# <<<< DO NOT MODIFY
# endregion
