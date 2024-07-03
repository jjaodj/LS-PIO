


from pigons import CosinePigeon
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
            neighbor.x()[i] += rand.uniform(-0.1, 0.1)  # Biến đổi nhỏ
            if neighbor.fitness() < best.fitness():
                best = copy.deepcopy(neighbor)
                improved = True
    return best


def tuba_search(pigeon):
    best = copy.deepcopy(pigeon)
    
    # Các tham số của Tuba Search
    tuba_length = 5  # Chiều dài của tuba
    frequency_range = (0.0, 1.0)  # Phạm vi tần số
    
    for _ in range(100):  # Số lượng lần lặp
        neighbor = copy.deepcopy(best)
        for i in range(len(neighbor.x())):
            frequency = frequency_range[0] + (frequency_range[1] - frequency_range[0]) * neighbor.x()[i]
            amplitude = 1  # Biên độ ban đầu
            for j in range(tuba_length):
                amplitude *= 0.5  # Giảm biên độ theo hàm mũ
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

        # Thêm bước tìm kiếm cục bộ
        gb = local_search(gb, hill_climbing)

        attr = gb.attr()
        print(t, " [tpr = ", f(gb.tpr), ", fpr = ", f(gb.fpr), "] a = ", attr, " (" + str(len(attr)) + "), ", "[tpr = ", f(pg.tpr), ", fpr = ", f(pg.fpr),
              "] a = ", len(pg.attr()))

        file.write(str(gb.fitness()) + '\t' + str(pg.fitness()) + '\r\n')

    print("--------------------------------")

    pop = list(pop)

    # Thực hiện các bước tiếp theo
    nnp = np // 2
    while nnp > 2:
        xc = SigmoidalPigeon.desirable_destination_center(pop, nnp)
        for i in range(0, nnp):
            pop[i].update_path(xc)
        pg = find_best(pop)
        if pg.fitness() < gb.fitness():
            gb = copy.deepcopy(pg)

        # Thêm bước tìm kiếm cục bộ
        gb = local_search(gb, hill_climbing)

        attr = gb.attr()
        print(" [tpr = ", f(gb.tpr), ", fpr = ", f(gb.fpr), "] a = ", attr, " (" + str(len(attr)) + "), ", "[tpr = ", f(pg.tpr),
              ", fpr = ", f(pg.fpr),
              "] a = ", len(pg.attr()))
        nnp = nnp // 2
        file.write(str(gb.fitness()) + '\t' + str(pg.fitness()) + '\r\n')

    # Thêm các bước cải tiến mới
    gb = find_best(pop)  # Bước 18: Xác định bồ câu tốt nhất (Xg)
    gb = local_search(gb, hill_climbing)  # Bước 19: Thực hiện tìm kiếm cục bộ
    gb = find_best([gb])  # Bước 20: Cập nhật bồ câu tốt nhất sau khi tìm kiếm cục bộ


    # Thực hiện vòng lặp cập nhật vận tốc và đường đi, huấn luyện Isolation Forest
    nc = number_of_iterations
    while nc >= 1:
        for p in pop:
            p.update_velocity_and_path(pg, nc)
            
        clf = train_lof(p) 
        for p in pop:
            
            p.evaluate_with_lof(clf)
    
        nc -= 1

    # Bước 30-36: Giảm kích thước quần thể và cập nhật vị trí bồ câu
    np = len(pop)
    while np >= 1:
        pop = sorted(pop, key=lambda p: p.fitness())
        np = np // 2
        xc = SigmoidalPigeon.desirable_destination_center(pop, np)
        for i in range(0, np):
            pop[i].update_path(xc)
        

        selected_attributes = gb.attr()
        print("Selected attributes:", selected_attributes)
# Thay đổi đường dẫn và tên file theo ý muốn của bạn
        output_filename = 'selected_attributes_tuba_if.csv'

# Ghi danh sách thuộc tính vào file CSV
        write_attributes_to_csv(selected_attributes, output_filename)

        acc, f_score = acc__f_score(gb.x())
        print('acc = ' + str(acc) + '\tf_score=' + str(f_score))
        file.write(str(gb.fitness()) + '\t' + str(gb.fitness()) + '\r\n')

    file.close()

init()
main()


'''


import csv
import random as rand
from math import exp
from pigons import CosinePigeon, SigmoidalPigeon  # Assuming both types are available
from problem import np, acc__f_score, init, number_of_iterations, R, get_number_of_inputs, L, U, calc_fitness, get_attr
import numpy
from sklearn.ensemble import IsolationForest
import copy

AW = .46
BW = .46
CW = .08

def f(value):
    return "{0:.3f}".format(value)

def find_best(pop):
    pg = None
    for p in pop:
        if (not pg) or (pg.fitness() > p.fitness()):
            pg = p
    return pg

def local_search(pigeon, method='hill_climbing'):
    # Implementing hill climbing for local search
    best_pigeon = copy.deepcopy(pigeon)
    for _ in range(10):  # Assuming a fixed number of local search iterations
        new_pigeon = copy.deepcopy(pigeon)
        new_pigeon.mutate(0.1)  # Assume mutate is a method that makes small changes to the pigeon
        if new_pigeon.fitness() < best_pigeon.fitness():
            best_pigeon = new_pigeon
    return best_pigeon

def write_attributes_to_csv(attributes, filename='selected_attributes.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Attribute'])
        for attr in attributes:
            writer.writerow([attr])

def main():
    file = open('fitness.txt', 'w')

    pop = set()

    for i in range(np):
        pop.add(SigmoidalPigeon(True))

    pg = find_best(pop)
    gb = copy.deepcopy(pg)

    file.write('global\tbest\r\n')
    file.write(str(gb.fitness()) + '\t' + str(pg.fitness()) + '\r\n')

    # First phase: Map and Compass Operator with Local Search
    for t in range(number_of_iterations):
        n_pop = set()
        for p in pop:
            p.update_velocity_and_path(pg, t)
            while p in n_pop:
                p.mutate(0.2)
            n_pop.add(p)

        pop = n_pop
        pg = find_best(pop)
        if pg.fitness() < gb.fitness():
            gb = copy.deepcopy(pg)
        
        # Local Search
        gb = local_search(gb, 'hill_climbing')

        file.write(str(gb.fitness()) + '\t' + str(pg.fitness()) + '\r\n')

    # Second phase: Landmark Operator
    pop = list(pop)
    nnp = np // 2
    while nnp > 2:
        xc = SigmoidalPigeon.desirable_destination_center(pop, nnp)
        for i in range(nnp):
            pop[i].update_path(xc)
        pg = find_best(pop)
        if pg.fitness() < gb.fitness():
            gb = copy.deepcopy(pg)
        nnp = nnp // 2
        file.write(str(gb.fitness()) + '\t' + str(pg.fitness()) + '\r\n')

    # Local Search at the end of second phase
    gb = local_search(gb, 'hill_climbing')

    # Third phase: Map and Compass Operator with Isolation Forest
    for t in range(number_of_iterations):
        n_pop = set()
        for p in pop:
            p.update_velocity_and_path(pg, t)
            while p in n_pop:
                p.mutate(0.2)
            n_pop.add(p)

        for p in pop:
            # Train Isolation Forest on the current pigeon's position
            isolation_forest = IsolationForest().fit([p.x()])
            # Evaluate fitness of the pigeon (assumed part of fitness calculation)
            p.evaluate_with_isolation_forest(isolation_forest)

        pop = n_pop
        pg = find_best(pop)
        if pg.fitness() < gb.fitness():
            gb = copy.deepcopy(pg)

        file.write(str(gb.fitness()) + '\t' + str(pg.fitness()) + '\r\n')

    # Fourth phase: Landmark Operator with Isolation Forest
    pop = list(pop)
    nnp = np // 2
    while nnp > 2:
        xc = SigmoidalPigeon.desirable_destination_center(pop, nnp)
        for i in range(nnp):
            pop[i].update_path(xc)
        pg = find_best(pop)
        if pg.fitness() < gb.fitness():
            gb = copy.deepcopy(pg)
        nnp = nnp // 2
        file.write(str(gb.fitness()) + '\t' + str(pg.fitness()) + '\r\n')

    acc, f_score = acc__f_score(gb.x())
    print('acc = '+str(acc) + '\tf_score=' + str(f_score))
    file.close()

    # Write selected attributes to CSV
    selected_attributes = gb.attr()  # Assuming `attr` method returns the selected attributes
    write_attributes_to_csv(selected_attributes)

init()
main()
'''