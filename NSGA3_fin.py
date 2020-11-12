# Import system modules
import arcpy
import numpy as np
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt

import random
import time

def optim(MU,NGEN):

    # load the shp of the scenario
    all_pts = "D:/04_PROJECTS/2001_WIND_OPTIM/B1_tmp.shp"
    #transform it to numpy array
    na = arcpy.da.TableToNumPyArray(all_pts, ['WT_ID', 'ENER_DENS', 'prod_MW'])

    # CXPB  is the probability with which two individuals
    #       are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.7, 0.4
    #MU,NGEN =20, 10
    enertarg = 4300000
    #some parameters to define the random individual
    nBITS = len(na)

    #production of energy
    sum_MW = np.sum(na['prod_MW'])

    low_targ = enertarg*0.96
    up_targ = enertarg * 1.06

    #the function to determine the initial random population which might reach the energy target
    def initial_indi():
     # relative to the total energy production to build the initial vector
     bound_up = (1.0 * up_targ / sum_MW)
     bound_low = (1.0 * low_targ / sum_MW)
     x3 = random.uniform(bound_low, bound_up)
     return np.random.choice([1, 0], size=(nBITS,), p=[x3, 1-x3])

    #some lists for the evaluation function
    enerd = list(na['ENER_DENS'])
    prod = list(na['prod_MW'])
    id = np.array(na['WT_ID'])

    #the evaluation function, taking the individual vector as input

    def evaluate(individual):
      individual = individual[0]
      #first check if the production of the seleced WT's is in the range between 4.31 and 4.29 TWH
      # goal 1
      mean_enerdsel = sum(x * y for x, y in zip(enerd, individual)) / sum(individual)
      # goal 2
      count_WTsel = sum(individual)
      # goal 3 (subset the input points by the WT_IDs which are in the ini pop (=1)
      WT_pop = np.column_stack((id, individual))
      WT_sel = WT_pop[WT_pop[:, [1]] == 1]
      WT_sel = WT_sel.astype(int)
      qry = '"WT_ID" IN ' + str(tuple(WT_sel))
      subset = arcpy.MakeFeatureLayer_management(all_pts, "tmp", qry)
      nn_output = arcpy.AverageNearestNeighbor_stats(subset, "EUCLIDEAN_DISTANCE", "NO_REPORT", "41290790000")
      clus = float(nn_output.getOutput(0))
      res = (clus, count_WTsel, mean_enerdsel)
      ## delete the feature tmp since otherwise it will not work in a loop
      arcpy.Delete_management("tmp")
      arcpy.Delete_management("subset")
      return(res)

    def feasible (individual):
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

    #initial individual and pop
    toolbox.register("initial_indi", initial_indi)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.initial_indi, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #evaluation and constraints
    toolbox.register("evaluate", evaluate)

    ##assign the feasibility of solutions and if not feasible a large number for the minimization tasks and a small number for the maximization task
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, (10e20, 10e20, 0)))

    #mate, mutate and select to perform crossover
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutPolynomialBounded,  low=0, up=1, eta=20, indpb=0.05)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)



    # initialize pareto front
    pareto = tools.ParetoFront(similar=np.array_equal)
    #hof = tools.HallOfFame(1)
        # Initialize statistics object
    #stats = tools.Statistics(lambda ind: ind.fitness.values)
    #stats.register("test", np.mean)
    #stats.register("test2", np.mean, axis=1)
    #stats.register("test3", np.mean, axis=2)
    #stats.register("std", np.std, axis=1)
    #stats.register("min", np.min, axis=1)
    #stats.register("max", np.max, axis=1)

   # first_stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    #third_stats = tools.Statistics(key=lambda ind: ind.fitness.values[2])
    #stats = tools.MultiStatistics(clus=first_stats)
    stats.register("min_WT", np.min, axis=0)
    #stats.register("max", np.max, axis=0)


    logbook = tools.Logbook()
    #logbook.header = "gen", "evals", "std", "min", "avg", "max"
    logbook.header = "gen", "evals", "min_WT"
    pop = toolbox.population(n=MU)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
   # invalid_ind = pop
    fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)



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

#select the next generation with NSGA3 from pop and offspring of size MU
        pop = toolbox.select(pop + offspring, MU)

        pareto.update(pop)

        # Compile statistics about the new population
        record = stats.compile(invalid_ind)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        print("--- %s seconds ---" % (time.time() - start_time))

    #plt.plot(logbook.select('gen'), logbook.select('min_WT'))
    #plt.show()
    return pop, logbook, pareto, invalid_ind