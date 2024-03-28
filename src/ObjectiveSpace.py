import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pymoo.indicators.hv import HV
import shapely.geometry as geom

np.random.seed(7)

lookup_hp = {
    'LightGCN': ['factors', 'n_layers', 'lr'],
    'NGCF': ['factors', 'n_layers', 'lr']
}

class ObjectivesSpace:
    def __init__(self, df, functions, model_name):
        self.model_name = model_name
        self.functions = functions
        self.df = df[df.columns.intersection(self._constr_obj())]
        self.points = self._get_points()

    def _constr_obj(self):
        objectives = list(self.functions.keys())
        objectives.insert(0, 'model')
        return objectives

    def _get_points(self):
        pts = self.df.to_numpy()
        factors = np.array(list(map(lambda x: 1 if x == 'max' else -1, list(self.functions.values()))))
        pts[:, 1:] = pts[:, 1:] * factors
        # sort points by decreasing sum of coordinates: the point having the greatest sum will be non dominated
        pts = pts[pts[:, 1:].sum(1).argsort()[::-1]]
        # initialize a boolean mask for non dominated and dominated points (in order to be contrastive)
        non_dominated = np.ones(pts.shape[0], dtype=bool)
        dominated = np.zeros(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            # process each point in turn
            n = pts.shape[0]
            # definition of Pareto optimality: for each point in the iteration, we find all points non dominated by
            # that point.
            mask1 = (pts[i + 1:, 1:] >= pts[i, 1:])
            mask2 = np.logical_not(pts[i + 1:, 1:] <= pts[i, 1:])
            non_dominated[i + 1:n] = (np.logical_and(mask1, mask2)).any(1)
            # A point could dominate another point, but it could also be dominated by a previous one in the iteration.
            # The following row take care of this situation by "keeping in memory" all dominated points in previous
            # iterations.
            dominated[i + 1:n] = np.logical_or(np.logical_not(non_dominated[i + 1:n]), dominated[i + 1:n])
        pts[:, 1:] = pts[:, 1:] * factors
        return pts[(np.logical_not(dominated))], pts[dominated]

    def get_nondominated(self):
        return pd.DataFrame(self.points[0], columns=self._constr_obj()).sort_values(by=list(self.functions.keys())[1])

    def get_dominated(self):
        return pd.DataFrame(self.points[1], columns=self._constr_obj())

    def get_nondominated_per_hp(self):
        temp_dict = {}
        return_dict = {}
        pts = self.get_nondominated()
        hps = lookup_hp[self.model_name]
        for hp in hps:
            pts[hp] = pts['model'].map(lambda x: x[x.find(hp) + len(str(hp) + '='):].split('_')[0])
            temp_dict[hp] = list(pts[hp].unique())
            return_dict[hp] = {}
        for hp in hps:
            hp_values = temp_dict[hp]
            for value in hp_values:
                return_dict[hp][value] = pts.loc[pts[hp] == value]
        return return_dict

    def get_dominated_per_hp(self):
        temp_dict = {}
        return_dict = {}
        pts = self.get_dominated()
        hps = lookup_hp[self.model_name]
        for hp in hps:
            pts[hp] = pts['model'].map(lambda x: x[x.find(hp) + len(str(hp) + '='):].split('_')[0])
            temp_dict[hp] = list(pts[hp].unique())
            return_dict[hp] = {}
        for hp in hps:
            hp_values = temp_dict[hp]
            for value in hp_values:
                return_dict[hp][value] = pts.loc[pts[hp] == value]
        return return_dict

    def plot(self, not_dominated, dominated, r):
        not_dominated = not_dominated.values
        dominated = dominated.values
        fig = plt.figure()
        if not_dominated.shape[1] == 3:
            ax = fig.add_subplot()
            ax.scatter(dominated[:, 1], dominated[:, 2], color='red')
            ax.scatter(not_dominated[:, 1], not_dominated[:, 2], color='blue')
            ax.scatter(r[0], r[1], color='green')
            plt.show()
        elif not_dominated.shape[1] == 4:
            ax = fig.add_subplot(projection='3d')
            ax.scatter(dominated[:, 1], dominated[:, 2], dominated[:, 3], color='red')
            ax.scatter(not_dominated[:, 1], not_dominated[:, 2], not_dominated[:, 3], color='blue')
            ax.scatter(r[0], r[1], r[2], color='green')
            ax.set_xlim3d(not_dominated[:, 1].min(), not_dominated[:, 1].max())
            ax.set_ylim3d(not_dominated[:, 2].min(), not_dominated[:, 2].max())
            ax.set_zlim3d(not_dominated[:, 3].min(), not_dominated[:, 3].max())
            plt.show()
        else:
            print("Cannot print >3-dimensional objective funtion space")

    def _get_distances(self, line_pts, pts):
        non_dom_line = line_pts.copy()
        all_pts = pts.copy()
        # non_dom = non_dom_pts.copy()
        # dom = dom_pts.copy()
        # pts = np.concatenate((non_dom, dom))
        # if normalization:
        #    for i in range(0, pts.shape[1]):
        #        non_dom[:, i] = (non_dom[:, i] - pts[:, i].min()) / (pts[:, i].max() - pts[:, i].min())
        #        dom[:, i] = (dom[:, i] - pts[:, i].min()) / (pts[:, i].max() - pts[:, i].min())
        #        non_dom_line[:, i] = (non_dom_line[:, i] - pts[:, i].min()) / (pts[:, i].max() - pts[:, i].min())

        distances = {}
        line = geom.LineString(non_dom_line[:, :][non_dom_line[:, 1].argsort()])

        # line = geom.LineString(self.points[0][:, 1:][self.points[0][:, 2].argsort()])
        i = 0
        for point in all_pts:
            distances[(i, tuple(point))] = geom.Point(point).distance(line)
            i += 1
        # for point in np.concatenate((self.points[1][:, 1:], self.points[0][:, 1:]), axis=0):
        #     distances[(i, tuple(point))] = geom.Point(point).distance(line)
        #     i += 1
        return distances

    def _minmax_normalization(self, pts, line_pts, all_pts):
        non_dom_line = line_pts.copy()
        all_pts = all_pts.copy()
        pts = pts.copy()
        # non_dom = non_dom.copy()
        # dom = dom.copy()
        # all_pts = all_pts.copy()
        # pts = np.concatenate((non_dom, dom))
        for i in range(0, all_pts.shape[1]):
            pts[:, i] = (pts[:, i] - all_pts[:, i].min()) / (all_pts[:, i].max() - all_pts[:, i].min())
            # non_dom[:, i] = (non_dom[:, i] - all_pts[:, i].min()) / (all_pts[:, i].max() - all_pts[:, i].min())
            # dom[:, i] = (dom[:, i] - all_pts[:, i].min()) / (all_pts[:, i].max() - all_pts[:, i].min())
            non_dom_line[:, i] = (non_dom_line[:, i] - all_pts[:, i].min()) / (all_pts[:, i].max() - all_pts[:, i].min())
        return non_dom_line, pts

    def mean_std(self, distances):
        mean = np.fromiter(distances.values(), dtype=float).mean()
        variance = ((np.fromiter(distances.values(), dtype=float) - mean) ** 2).sum() / (
                np.fromiter(distances.values(), dtype=float).shape[0] - 1)
        standard_deviation = variance ** (1 / 2)
        return standard_deviation, mean

    def get_statistics(self, normalization=True):
        non_dom, dom = self.points[0][:, 1:].astype('float'), self.points[1][:, 1:].astype('float')
        pts = np.concatenate((self.points[0][:, 1:].astype('float'), self.points[1][:, 1:].astype('float')))
        if normalization:
            line_pts, all_pts = self._minmax_normalization(pts, non_dom, np.concatenate((non_dom, dom)))
            distances = self._get_distances(line_pts, all_pts)
        else:
            distances = self._get_distances(non_dom, pts)
        # distances = self._get_distances(non_dom, non_dom, dom)
        return self.mean_std(distances)

        # mean = np.fromiter(distances.values(), dtype=float).mean()
        # variance = ((np.fromiter(distances.values(), dtype=float) - mean) ** 2).sum() / (
        #             np.fromiter(distances.values(), dtype=float).shape[0] - 1)
        # standard_deviation = variance ** (1 / 2)
        # return standard_deviation, mean

    def get_statistics_per_hp(self, normalization=True):
        non_dom, dom = self.points[0][:, 1:].astype('float'), self.points[1][:, 1:].astype('float')
        non_dom_hp, dom_hp = self.get_nondominated_per_hp(), self.get_dominated_per_hp()
        hps = lookup_hp[self.model_name]
        stats_hp = {}
        for hp in hps:
            stats_hp[hp] = {}
            values = set()
            for el in dom_hp[hp].keys():
                values.add(el)
            for el in non_dom_hp[hp].keys():
                values.add(el)
            for value in values:
                try:
                    pts = np.concatenate((non_dom_hp[hp][value].values[:, 1:3].astype('float'), dom_hp[hp][value].values[:, 1:3].astype('float')))
                except KeyError:
                    try:
                        pts = non_dom_hp[hp][value].values[:, 1:3].astype('float')
                    except KeyError:
                        pts = dom_hp[hp][value].values[:, 1:3].astype('float')
                if normalization:
                    line_pts, all_pts = self._minmax_normalization(pts, non_dom, np.concatenate((non_dom, dom)))
                    distances = self._get_distances(line_pts, all_pts)
                else:
                    distances = self._get_distances(non_dom, pts)
                stats_hp[hp][value] = self.mean_std(distances)
        return stats_hp

    """
        @For: Spread
        @Output: Measures the range of a solution set
        @Tips: Higher the value, better extensity 
    """
    def maximum_spread(self):
        n_objs = self.points[0].shape[1]
        ms = 0
        for j in range(1, n_objs):
            ms += (max(self.points[0][:, j]) - min(self.points[0][:, j]))**2
        return np.sqrt(ms)

    """
        @For: Uniformity
        @Output: Measures the variation of the distance between solutions in a set.
        @Tips: lower the value, better the uniformity 
        @From: https://github.com/Valdecy/pyMultiobjective/blob/main/pyMultiobjective/util/indicators.py
    """
    def spacing(self):
        sol = np.copy(self.points[0][:, 1:])
        dm = np.zeros(sol.shape[0])
        for i in range(0, sol.shape[0]):
            try:
                dm[i] = min([np.linalg.norm(sol[i] - sol[j]) for j in range(0, sol.shape[0]) if i != j])
            except ValueError:
                return 0
        d_mean = np.mean(dm)
        spacing = np.sqrt(np.sum((dm - d_mean) ** 2) / sol.shape[0])
        return spacing

    """
        @For: Cardinality
        @Output: Considers the proportion of non-dominated solutions
        @Tips: Smaller value is preferable
    """
    def error_ratio(self):
        return self.points[0].shape[0] / (self.points[0].shape[0] + self.points[1].shape[0])

    def _get_pareto(self, sets):
        pts = sets.to_numpy()
        factors = np.array(list(map(lambda x: 1 if x == 'max' else -1, list(self.functions.values()))))
        pts[:, 1:] = pts[:, 1:] * factors
        # sort points by decreasing sum of coordinates: the point having the greatest sum will be non dominated
        pts = pts[pts[:, 1:].sum(1).argsort()[::-1]]
        # initialize a boolean mask for non dominated and dominated points (in order to be contrastive)
        non_dominated = np.ones(pts.shape[0], dtype=bool)
        dominated = np.zeros(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            # process each point in turn
            n = pts.shape[0]
            # definition of Pareto optimality: for each point in the iteration, we find all points non dominated by
            # that point.
            mask1 = (pts[i + 1:, 1:] >= pts[i, 1:])
            mask2 = np.logical_not(pts[i + 1:, 1:] <= pts[i, 1:])
            non_dominated[i + 1:n] = (np.logical_and(mask1, mask2)).any(1)
            # A point could dominate another point, but it could also be dominated by a previous one in the iteration.
            # The following row take care of this situation by "keeping in memory" all dominated points in previous
            # iterations.
            dominated[i + 1:n] = np.logical_or(np.logical_not(non_dominated[i + 1:n]), dominated[i + 1:n])
        pts[:, 1:] = pts[:, 1:] * factors
        return pts[(np.logical_not(dominated))]

    """
            @For: Convergence
            @Type: Dominance-based QI
            @Output: Measures the relative quality between two sets on covergence and cardinality
    """
    def c_indicator(self, set_a):
        set_b = self.get_nondominated()
        sets = pd.concat([set_a, set_b], axis=0)
        not_dom = pd.DataFrame(self._get_pareto(sets), columns=self._constr_obj())
        c_ind = pd.merge(not_dom, set_b, how='inner', on=['model']).shape[0] / set_b.shape[0]
        return c_ind

    """
        @For: All Quality Aspects
        @Type: Volume-based
        @Output: A set that achieves the maximum HV value for a given problem will contain all Pareto optimal solutions.
    """
    def hypervolumes(self, r):
        factors = np.array(list(map(lambda x: -1 if x == 'max' else 1, list(self.functions.values()))))
        hv_pts = np.copy(self.points[0])
        hv_pts[:, 1:] = hv_pts[:, 1:] * factors
        r = r * factors
        not_dominated = pd.DataFrame(hv_pts, columns=self._constr_obj())
        x = not_dominated[list(self.functions.keys())]
        ind = HV(ref_point=r)
        return ind(np.array(x.values))