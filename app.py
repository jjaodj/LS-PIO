
from pigons import SigmoidalPigeon
from problem import np, acc__f_score
from problem import init
from problem import number_of_iterations
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import copy
import random as rand
import csv
import math

def f(value):
    return "{0:.3f}".format(value)


def find_best(pop):
    pg = None
    for p in pop:
        if (not pg) or (pg.fitness() > p.fitness()):
            pg = p
    return pg

def write_attributes_to_csv(attributes, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for attr in attributes:
            writer.writerow([attr])

def hill_climbing(pigeon):
    best = copy.deepcopy(pigeon)
    improved = True
    while improved:
        improved = False
        neighbor = copy.deepcopy(best)
        for i in range(len(neighbor.x())):
            neighbor.x()[i] += rand.uniform(-0.1, 0.1)  
            if neighbor.fitness() < best.fitness():
                best = copy.deepcopy(neighbor)
                improved = True
    return best


def tuba_search(pigeon):
    best = copy.deepcopy(pigeon)
    
   
    tuba_length = 5  
    frequency_range = (0.0, 1.0)  
    
    for _ in range(100):  
        neighbor = copy.deepcopy(best)
        for i in range(len(neighbor.x())):
            frequency = frequency_range[0] + (frequency_range[1] - frequency_range[0]) * neighbor.x()[i]
            amplitude = 1  
            for j in range(tuba_length):
                amplitude *= 0.5  
                neighbor.x()[i] += amplitude * math.sin(2 * math.pi * frequency)
            if neighbor.fitness() < best.fitness():
                best = copy.deepcopy(neighbor)
    
    return best

def local_search(pigeon, search_method):
    return tuba_search(pigeon)

def train_isolation_forest(pigeon):
    X = [pigeon.x()]
    clf = IsolationForest(random_state=0).fit(X)
    return clf

def train_lof(pop):
    X = [p.x() for p in pop]
    clf = LocalOutlierFactor().fit(X)
    return clf

def train_one_class_svm(pigeon):
    X = [pigeon.x()]
    clf = OneClassSVM(nu=0.1, kernel="rbf", gamma="auto").fit(X)
    return clf

def main():
    file = open('fitness.txt', 'w')

    pop = set()
    np=64

    for i in range(0, np):
        pop.add(SigmoidalPigeon(True))

    pg = find_best(pop)
    gb = copy.deepcopy(pg)

    file.write('global\tbest\r\n')
    file.write(str(gb.fitness()) + '\t' + str(pg.fitness()) + '\r\n')

    for t in range(0, number_of_iterations):
        n_pop = set()
        for p in pop:
            p.update_velocity_and_path(pg, t)
            while p in n_pop:
                p.mutate(0.2)
            n_pop.add(p)

        while len(n_pop) < np:
            n_pop.add(SigmoidalPigeon(True))
            print('Error')

        pop = n_pop
        pg = find_best(pop)
        if pg.fitness() < gb.fitness():
            gb = copy.deepcopy(pg)

     
        gb = local_search(gb, hill_climbing)

        attr = gb.attr()
        print(t, " [tpr = ", f(gb.tpr), ", fpr = ", f(gb.fpr), "] a = ", attr, " (" + str(len(attr)) + "), ", "[tpr = ", f(pg.tpr), ", fpr = ", f(pg.fpr),
              "] a = ", len(pg.attr()))

        file.write(str(gb.fitness()) + '\t' + str(pg.fitness()) + '\r\n')

    print("--------------------------------")

    pop = list(pop)

    nnp = np // 2
    while nnp > 2:
        xc = SigmoidalPigeon.desirable_destination_center(pop, nnp)
        for i in range(0, nnp):
            pop[i].update_path(xc)
        pg = find_best(pop)
        if pg.fitness() < gb.fitness():
            gb = copy.deepcopy(pg)

  
        gb = local_search(gb, hill_climbing)

        attr = gb.attr()
        print(" [tpr = ", f(gb.tpr), ", fpr = ", f(gb.fpr), "] a = ", attr, " (" + str(len(attr)) + "), ", "[tpr = ", f(pg.tpr),
              ", fpr = ", f(pg.fpr),
              "] a = ", len(pg.attr()))
        nnp = nnp // 2
        file.write(str(gb.fitness()) + '\t' + str(pg.fitness()) + '\r\n')

    
    gb = find_best(pop)  
    gb = local_search(gb, hill_climbing) 
    gb = find_best([gb]) 


    
    nc = number_of_iterations
    while nc >= 1:
        for p in pop:
            p.update_velocity_and_path(pg, nc)
            
        clf = train_lof(p) 
        for p in pop:
            
            p.evaluate_with_lof(clf)
    
        nc -= 1


    np = len(pop)
    while np >= 1:
        pop = sorted(pop, key=lambda p: p.fitness())
        np = np // 2
        xc = SigmoidalPigeon.desirable_destination_center(pop, np)
        for i in range(0, np):
            pop[i].update_path(xc)
        

        selected_attributes = gb.attr()
        print("Selected attributes:", selected_attributes)

        output_filename = 'selected_attributes_tuba_if.csv'


        write_attributes_to_csv(selected_attributes, output_filename)

        acc, f_score = acc__f_score(gb.x())
        print('acc = ' + str(acc) + '\tf_score=' + str(f_score))
        file.write(str(gb.fitness()) + '\t' + str(gb.fitness()) + '\r\n')

    file.close()

init()
main()


