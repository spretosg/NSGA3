# Import system modules
import arcpy
from arcpy import  env
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap.tools._hypervolume import pyhv as hv

import random
import time
path = "D:/04_PROJECTS/2001_WIND_OPTIM/WIND_OPTIM_git/intermediate_steps/3_wp/NSGA3_RES/in/B3_FFF+.shp"
def optim(MU, NGEN, path):
    arcpy.env.overwriteOutput = True
    #load pts in memory
    all_pts = r"in_memory/inMemoryFeatureClass"
    in_pts = path
    arcpy.CopyFeatures_management(in_pts, all_pts)

    #transform it to numpy array
    na = arcpy.da.TableToNumPyArray(all_pts, ['WT_ID', 'ENER_DENS', 'prod_MW'])

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.7, 0.4
    #MU,NGEN =20, 10
    enertarg = 4300000
    #some parameters to define the random individual
    nBITS = len(na)
    #total production of energy
    sum_MW = np.sum(na['prod_MW'])

    #the 4.3TWh/y represent the minimal target to reach and app. 4.6Twh is the upper bandwidth
    low_targ = enertarg
    up_targ = enertarg * 1.07

    #the function to determine the initial random population which might reach the energy target bandwidth
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
        #select the next generation with NSGA3 from pop and offspring of size MU
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

    #fitness pareto
    fitness_pareto = toolbox.map(toolbox.evaluate, pareto)
    fitness_pareto = np.array(fitness_pareto)
    fitness_pareto = {'CLUS': fitness_pareto[:, 0], 'N_WT': fitness_pareto[:, 1], 'ENERDENS': fitness_pareto[:, 2]}
    #pareto items and robustness
    par_items = np.array(pareto.items)
    par_rob = np.array(1.0 * sum(par_items[1:len(par_items)]) / len(par_items))
    par_rob = par_rob.ravel()
    par_rob_mat = np.column_stack((id, par_rob))
    par_rob_mat = {'WT_ID2': par_rob_mat[:, 0], 'par_rob': par_rob_mat[:, 1]}

    #logbook
    gen = np.array(logbook1.select('gen'))
    TIME = np.array(logbook1.select('TIME'))
    WT = np.array(logbook2.select('min_WT'))
    clus = np.array(logbook1.select('min_clus'))
    enerd = np.array(logbook3.select('max_enerd'))
    logbook = np.column_stack((gen, TIME, WT, clus, enerd))
    logbook = {'GENERATION': logbook[:, 0], 'TIME': logbook[:, 1], 'N_WT': logbook[:, 2], 'CLUS': logbook[:, 3],
               'ENERDENS': logbook[:, 4]}

    return HV, par_rob_mat, fitness_pareto, logbook
