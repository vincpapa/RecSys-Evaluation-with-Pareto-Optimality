from src.ObjectiveSpace import *
import pandas as pd
import numpy as np
from functools import reduce
import glob
import os

lookup_hp = {
    'LightGCN': ['factors', 'n_layers', 'lr'],
    'NGCF': ['factors', 'n_layers', 'lr']
}

if __name__ == '__main__':
    """
    @dataset: amazon_music, goodreads, movielens1m
    @scenario1:
        obj1 = 'nDCG'
        opt1 = 'max'
        obj2 = 'Gini'
        opt2 = 'max'
        obj3 = 'EPC'
        opt3 = 'max' #Add obj3 and opt3 to the current code - line 26 - line 
    @scenario2: 
        obj1 = 'nDCG'
        opt1 = 'max'
        obj2 = 'APLT'
        opt2 = 'max'
    """
    model_name = 'LightGCN'
    dataset = 'amazon_books'
    dir = os.listdir(f'data/{dataset}/{model_name}')
    obj1 = 'nDCG'
    opt1 = 'max'
    obj2 = 'APLT'
    opt2 = 'max'
    obj3 = ''
    opt3 = ''
    """
    @scenario1:
        reference_point = np.array([0, 0, 0])
    @scenario2:
        reference_point = np.array([0, 0])
    """
    reference_point = np.array([0, 0])
    results = []
    hypervolumes = []
    for element in dir:
        model = pd.read_csv(f'data/{dataset}/{model_name}/{element}', sep='\t')
        obj = ObjectivesSpace(model, {obj1: opt1, obj2: opt2}, model_name, )
        print('****** OPTIMAL *****')
        print(obj.get_nondominated())
        non_dominated = obj.get_nondominated()
        non_dominated_hp = obj.get_nondominated_per_hp()
        for k, v in non_dominated_hp.items():
            for i, j in v.items():
                j.to_csv(f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_{obj3}_{k}={i}_not_dominated.csv', sep=',', index=False)
        non_dominated.to_csv(f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_{obj3}_not_dominated.csv', sep=',', index=False)
        print('****** DOMINATED *****')
        print(obj.get_dominated())
        dominated_hp = obj.get_dominated_per_hp()
        for k, v in dominated_hp.items():
            for i, j in v.items():
                j.to_csv(f'results/{dataset}/{model_name}/{element[4:-4]}_{obj1}_{obj2}_{obj3}_{k}={i}_dominated.csv', sep=',',
                         index=False)

        obj.plot(obj.get_nondominated(), obj.get_dominated(), reference_point)
        ms = obj.maximum_spread()
        sp = obj.spacing()
        er = obj.error_ratio()
        hv = obj.hypervolumes(reference_point)
        c = non_dominated.shape[0]
        hv_c = hv / c
        print(ms, sp, er, hv, c, hv_c)
        results.append([element[4:-4], ms, sp, er, hv, c, hv_c])
        print(obj.get_statistics())
        print(obj.get_statistics_per_hp())
        print(obj.get_statistics(False))
        # print(obj.get_nondominated_per_hp())
    res_df = pd.DataFrame(results, columns=['model', 'MS', 'SP', 'ER', 'HV', 'C', 'HV/C'])
    print('')
    pass