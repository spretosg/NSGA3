# Import system modules
import arcpy
from arcpy import env
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap.tools._hypervolume import pyhv as hv
#import scipy.spatial

from scipy.spatial import cKDTree

import random
import time


def optim(MU, NGEN, path,CXPB, MUTPB):
    fc = path
    na = arcpy.da.FeatureClassToNumPyArray(fc, ["WT_ID", "ENER_DENS", "prod_MW", "SHAPE@XY"], explode_to_points=True)

    ##here we calculate the expected nearest neighbor distance (in meters) of the scenario
    nBITS = len(na)

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    #CXPB, MUTPB = 0.8, 0.6
    # MU,NGEN =20, 10
    enertarg = 4300000
    # some parameters to define the random individual

    # total production of energy
    sum_MW = np.sum(na['prod_MW'])

    # the 4.3TWh/y represent the minimal target to reach and app. 4.6Twh is the upper bandwidth
    low_targ = enertarg
    up_targ = enertarg * 1.07

    # the function to determine the initial random population which might reach the energy target bandwidth
    def initial_indi():
        # relative to the total energy production to build the initial vector
        bound_up = (1.0 * up_targ / sum_MW)
        bound_low = (1.0 * low_targ / sum_MW)
        x3 = random.uniform(bound_low, bound_up)
        return np.random.choice([1, 0], size=(nBITS,), p=[x3, 1 - x3])

    # some lists for the evaluation function
    enerd = list(na['ENER_DENS'])
    prod = list(na['prod_MW'])
    id = np.array(na['WT_ID'])
    _xy = list(na['SHAPE@XY'])

    # the evaluation function, taking the individual vector as input

    def evaluate(individual):
        individual = individual[0]
        prod_MWsel = sum(x * y for x, y in zip(prod, individual))
        #check if the total production is witin boundaries, if not return a penalty vector
        if up_targ >= prod_MWsel >= low_targ:
            # goal 1
            mean_enerdsel = sum(x * y for x, y in zip(enerd, individual)) / sum(individual)
            # goal 2
            count_WTsel = sum(individual)
            # goal 3 zip the individual vector to the _xy coordinates
            subset = np.column_stack((_xy,individual))
            #subset the data that only the 1 remains
            subset = subset[subset[:, 2] == 1]
            subset = np.delete(subset, 2, 1)
            tree = cKDTree(subset)
            dists = tree.query(subset, 2)
            nn_dist = dists[0][:, 1]
            rE = 1 / (2 * math.sqrt(1.0 * len(subset) / 41290790000))
            rA= np.mean(nn_dist)
            clus = rA/rE
            res = (clus, count_WTsel, mean_enerdsel)
            ## delete the feature tmp since otherwise it will not work in a loop
            arcpy.Delete_management("tmp")
            arcpy.Delete_management("subset")
        else:
            res = (10e20, 10e20, 0)
        return res

    #def feasible(individual):
        individual = individual[0]
        prod_MWsel = sum(x * y for x, y in zip(prod, individual))
        if (prod_MWsel <= up_targ and prod_MWsel >= low_targ):
            return True
        return False

    ### setup NSGA3 with deap (minimize the first two goals returned by the evaluate function and maximize the third one)
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    ref_points = tools.uniform_reference_points(nobj=3, p=12)
    ##setup the optim toolbox I do not understand that totally
    toolbox = base.Toolbox()
    # initial individual and pop
    toolbox.register("initial_indi", initial_indi)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.initial_indi, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # evaluation and constraints
    toolbox.register("evaluate", evaluate)
    ##assign the feasibility of solutions and if not feasible a large number for the minimization tasks and a small number for the maximization task
    #toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, (10e20, 10e20, 0)))
    # mate, mutate and select to perform crossover
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    ### initialize pareto front
    pareto = tools.ParetoFront(similar=np.array_equal)
    ### initialize population
    pop = toolbox.population(n=MU)

    first_stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    second_stats = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    third_stats = tools.Statistics(key=lambda ind: ind.fitness.values[2])

    first_stats.register("min_clus", np.min, axis=0)
    second_stats.register("min_WT", np.min, axis=0)
    third_stats.register("max_enerd", np.max, axis=0)

    logbook1 = tools.Logbook()
    logbook2 = tools.Logbook()
    logbook3 = tools.Logbook()
    logbook1.header = "gen", "evals", "TIME", "min_clus"
    logbook2.header = "gen", "evals", "min_WT"
    logbook2.header = "gen", "evals", "max_enerd"

    HV = []
    # Evaluate the initial individuals with an invalid fitness
    print("-- fitness of initial population --")
    start_time = time.time()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    ## Hyper volume of initial fitness (scale the n_WT and change the value of the maximization goal with -1
    fitness_trans = np.array(fitnesses)
    fitness_trans[:, 1] *= 1.0 / nBITS
    fitness_trans[:, 2] *= -1
    hyp = hv.hypervolume(fitness_trans, ref=np.array([1, 1, 1]))
    HV.append(hyp)

    end_time = time.time()
    delt_time = end_time - start_time

    record1 = first_stats.compile(pop)
    logbook1.record(gen=0, evals=len(invalid_ind), TIME=delt_time, **record1)

    record2 = second_stats.compile(pop)
    logbook2.record(gen=0, evals=len(invalid_ind), **record2)

    record3 = third_stats.compile(pop)
    logbook3.record(gen=0, evals=len(invalid_ind), **record3)

    # Begin the evolution with NGEN repetitions
    for gen in range(1, NGEN):
        print("-- Generation %i --" % gen)
        start_time = time.time()
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1[0], child2[0])
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant[0])
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        fitness_trans = np.array(fitnesses)
        fitness_trans[:, 1] *= 1.0 / nBITS
        fitness_trans[:, 2] *= -1
        ## Hyper volume
        hyp = hv.hypervolume(fitness_trans, ref=np.array([1, 1, 1]))
        HV.append(hyp)
        # select the next generation with NSGA3 from pop and offspring of size MU
        pop = toolbox.select(pop + offspring, MU)
        pareto.update(pop)

        record1 = first_stats.compile(invalid_ind)
        logbook1.record(gen=gen, evals=len(invalid_ind), TIME=delt_time, **record1)

        record2 = second_stats.compile(invalid_ind)
        logbook2.record(gen=gen, evals=len(invalid_ind), **record2)

        record3 = third_stats.compile(invalid_ind)
        logbook3.record(gen=gen, evals=len(invalid_ind), **record3)

        end_time = time.time()
        delt_time = end_time - start_time
        print("--- %s seconds ---" % delt_time)

    # fitness pareto
    fitness_pareto = toolbox.map(toolbox.evaluate, pareto)
    fitness_pareto = np.array(fitness_pareto)
    fitness_pareto = {'CLUS': fitness_pareto[:, 0], 'N_WT': fitness_pareto[:, 1], 'ENERDENS': fitness_pareto[:, 2]}
    # pareto items and robustness
    par_items = np.array(pareto.items)
    par_rob = np.array(1.0 * sum(par_items[1:len(par_items)]) / len(par_items))
    par_rob = par_rob.ravel()
    par_rob_mat = np.column_stack((id, par_rob))
    par_rob_mat = {'WT_ID2': par_rob_mat[:, 0], 'par_rob': par_rob_mat[:, 1]}

    # logbook
    gen = np.array(logbook1.select('gen'))
    TIME = np.array(logbook1.select('TIME'))
    WT = np.array(logbook2.select('min_WT'))
    clus = np.array(logbook1.select('min_clus'))
    enerd = np.array(logbook3.select('max_enerd'))
    logbook = np.column_stack((gen, TIME, WT, clus, enerd))
    logbook = {'GENERATION': logbook[:, 0], 'TIME': logbook[:, 1], 'N_WT': logbook[:, 2], 'CLUS': logbook[:, 3],
               'ENERDENS': logbook[:, 4]}

    return HV, par_rob_mat, fitness_pareto, logbook
