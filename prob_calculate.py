import math
import numpy as np

class WinGameProCalculator:
    def __init__(self, m, p):
        self.m = m
        self.p = p
        self.key_p = p ** 2 / (1 - 2 * p * (1 - p))
        self.results = {}

    def GetProbability(self, i, j):
        if i >= self.m and i >= j + 2:
            return 1
        if j >= self.m and j >= i + 2:
            return 0
        if i == j and i >= self.m - 2:
            return self.key_p
        if (i, j) in self.results:
            return self.results[(i, j)]

        self.results[(i, j)] = self.p * self.GetProbability(i + 1, j) + (1 - self.p) * self.GetProbability(i, j + 1)
        return self.results[(i, j)]

    def UpdateProbability(self, p):
        self.p = p
        self.key_p = p ** 2 / (1 - 2 * p * (1 - p))
        self.results = {}

    def CalcProbability4(self):
        q = 1 - self.p
        self.results[(0, 0)] = self.p ** 6 + 6 * self.p ** 5 * q + 15 * self.p ** 4 * q ** 2 + 20 * self.p ** 3 * q ** 3 * (self.p ** 2 / (1 - 2 * self.p * q))
        return self.results[(0, 0)]


class WinSetProCalculator:
    def __init__(self, m, p1, p2, T):
        self.m = m
        self.p1 = p1
        self.p2 = p2
        self.T = T
        self.key_p = p1 * p2 / (1 - (p1 * (1 - p2) + p2 * (1 - p1)))
        self.results = {}

    def GetProbability_func(self, i, j):
        if i >= self.m and i >= j + 2:
            return 1
        if j >= self.m and j >= i + 2:
            return 0
        if i == j and i >= self.m - 2:
            return self.key_p
        if (i, j) in self.results:
            return self.results[(i, j)]

        if self.T == 0 :
            self.T = 1
            self.results[(i, j)] = self.p1 * self.GetProbability_func(i + 1, j) + (1 - self.p1) * self.GetProbability_func(i, j + 1)
            return self.results[(i, j)]
        else:
            self.T = 0
            self.results[(i, j)] = self.p2 * self.GetProbability_func(i + 1, j) + (1 - self.p2) * self.GetProbability_func(i, j + 1)
            return self.results[(i, j)]

    def GetProbability(self, i, j, T):
        self.T = T
        if T != self.T:
            self.results= {}

        result = self.GetProbability_func(i, j)
        self.T = T
        return result

    def UpdateProbability(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.key_p = p1 * p2 / (1 - (p1 * (1 - p2) + p2 * (1 - p1)))
        self.results = {}

    def UpdateT(self, T):
        self.T = T
        self.results = {}


class WinMatchProCalculator:
    def __init__(self, m, p):
        self.m = m
        self.p = p
        self.results = {}

    def GetProbability(self, i, j):
        if i >= self.m:
            return 1
        if j >= self.m:
            return 0
        if (i, j) in self.results:
            return self.results[(i, j)]

        self.results[(i, j)] = self.p * self.GetProbability(i + 1, j) + (1 - self.p) * self.GetProbability(i, j + 1)
        return self.results[(i, j)]

    def UpdateProbability(self, p):
        self.p = p
        self.results = {}


class MomentumCaculater:
    def __init__(self, p1, p2, T, a):
        self.a = a
        self.p1 = p1
        self.p2 = p2
        self.T = T
        self.Leverages=[]
        self.wingame1 = self.wingame2 = self.winset1 = self.winset2 = 0
        self.Game1_P_caculator = WinGameProCalculator(4, self.p1)
        self.Game1_win_P = self.Game1_P_caculator.GetProbability(0, 0)
        self.Game2_P_caculator = WinGameProCalculator(4, self.p2)
        self.Game2_win_P = self.Game2_P_caculator.GetProbability(0, 0)
        self.set_P_caculator = WinSetProCalculator(7, self.Game2_win_P, self.Game2_win_P, T)
        self.set_win_P = self.set_P_caculator.GetProbability(0, 0, 0)
        self.match_P_caculator = WinMatchProCalculator(3, self.set_win_P)

    def PredictPro(self, i, j, T):
        if T == 0:
            Game1_win_P = self.Game1_P_caculator.GetProbability(i, j)
            Game_win_Set_win_P = self.set_P_caculator.GetProbability(self.wingame1+1, self.wingame2, 1)
            Game_loss_Set_win_P = self.set_P_caculator.GetProbability(self.wingame1, self.wingame2+1, 1)
            Set_win_Match_win_P = self.match_P_caculator.GetProbability(self.winset1+1, self.winset2)
            Set_loss_Match_win_P = self.match_P_caculator.GetProbability(self.winset1, self.winset2+1)

            Set_win_P = Game1_win_P * Game_win_Set_win_P + (1 - Game1_win_P) * Game_loss_Set_win_P
            Match_win_P = Set_win_P * Set_win_Match_win_P + (1 - Set_win_P) * Set_loss_Match_win_P

            result = Match_win_P
            return result
        else:
            Game2_win_P = self.Game1_P_caculator.GetProbability(i, j)
            Game_win_Set_win_P = self.set_P_caculator.GetProbability(self.wingame1 + 1, self.wingame2, 1)
            Game_loss_Set_win_P = self.set_P_caculator.GetProbability(self.wingame1, self.wingame2 + 1, 1)
            Set_win_Match_win_P = self.match_P_caculator.GetProbability(self.winset1 + 1, self.winset2)
            Set_loss_Match_win_P = self.match_P_caculator.GetProbability(self.winset1, self.winset2 + 1)

            Set_win_P = Game2_win_P * Game_win_Set_win_P + (1 - Game2_win_P) * Game_loss_Set_win_P
            Match_win_P = Set_win_P * Set_win_Match_win_P + (1 - Set_win_P) * Set_loss_Match_win_P

            result = Match_win_P
            return result

    def GetLeverage(self, i, j, T, win):
        if T == 0:
            Point_win_Game1_win_P = self.Game1_P_caculator.GetProbability(i+1, j)
            Point_loss_Game1_win_P = self.Game1_P_caculator.GetProbability(i, j+1)
            Game_win_Set_win_P = self.set_P_caculator.GetProbability(self.wingame1+1, self.wingame2, 1)
            Game_loss_Set_win_P = self.set_P_caculator.GetProbability(self.wingame1, self.wingame2+1, 1)
            Set_win_Match_win_P = self.match_P_caculator.GetProbability(self.winset1+1, self.winset2)
            Set_loss_Match_win_P = self.match_P_caculator.GetProbability(self.winset1, self.winset2+1)

            Point_win_Set_win_P = Point_win_Game1_win_P * Game_win_Set_win_P + (1 - Point_win_Game1_win_P) * Game_loss_Set_win_P
            Point_loss_Set_win_P = Point_loss_Game1_win_P * Game_win_Set_win_P + (1 - Point_loss_Game1_win_P) * Game_loss_Set_win_P

            Point_win_Match_win_P = Point_win_Set_win_P * Set_win_Match_win_P + (1 - Point_win_Set_win_P) * Set_loss_Match_win_P
            Point_loss_Match_win_P = Point_loss_Set_win_P * Set_win_Match_win_P + (1 - Point_loss_Set_win_P) * Set_loss_Match_win_P

            if(win):
                result = Point_win_Match_win_P - Point_loss_Match_win_P
            else:
                result = -(Point_win_Match_win_P - Point_loss_Match_win_P)

            self.Leverages.append(result)
            return result
        else:
            Point_win_Game2_win_P = self.Game2_P_caculator.GetProbability(i+1, j)
            Point_loss_Game2_win_P = self.Game2_P_caculator.GetProbability(i, j+1)
            Game_win_Set_win_P = self.set_P_caculator.GetProbability(self.wingame1+1, self.wingame2, 0)
            Game_loss_Set_win_P = self.set_P_caculator.GetProbability(self.wingame1, self.wingame2 + 1, 0)
            Set_win_Match_win_P = self.match_P_caculator.GetProbability(self.winset1+1, self.winset2)
            Set_loss_Match_win_P = self.match_P_caculator.GetProbability(self.winset1, self.winset2+1)

            Point_win_Set_win_P = Point_win_Game2_win_P * Game_win_Set_win_P + (1 - Point_win_Game2_win_P) * Game_loss_Set_win_P
            Point_loss_Set_win_P = Point_loss_Game2_win_P * Game_win_Set_win_P + (1 - Point_loss_Game2_win_P) * Game_loss_Set_win_P

            Point_win_Match_win_P = Point_win_Set_win_P * Set_win_Match_win_P + (1 - Point_win_Set_win_P) * Set_loss_Match_win_P
            Point_loss_Match_win_P = Point_loss_Set_win_P * Set_win_Match_win_P + (1 - Point_loss_Set_win_P) * Set_loss_Match_win_P

            if (win):
                result = Point_win_Match_win_P - Point_loss_Match_win_P
            else:
                result = -(Point_win_Match_win_P - Point_loss_Match_win_P)

            self.Leverages.append(result)
            return result

    def GetMomentum(self):
        t = len(self.Leverages)
        denominator = sum([(1 - self.a) ** i for i in range(t)])
        weighted_sum = sum([(1 - self.a) ** i * self.Leverages[t-i-1] for i in range(t)])

        result = weighted_sum / denominator

        return result, t

    def UpdateProbability(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.Game1_P_caculator.UpdateProbability(p1)
        self.Game1_win_P = self.Game1_P_caculator.GetProbability(0, 0)
        self.Game2_P_caculator.UpdateProbability(p2)
        self.Game2_win_P = self.Game2_P_caculator.GetProbability(0, 0)
        self.set_P_caculator.UpdateProbability(self.Game1_win_P, self.Game2_win_P)
        self.set_win_P = self.set_P_caculator.GetProbability(0, 0, 0)
        self.match_P_caculator = WinMatchProCalculator(3, self.set_win_P)


    def UpdateT(self, T):
        self.T = T
        self.set_P_caculator.UpdateT(T)

    def UpdateSet(self, set1, set2):
        self.winset1 = set1
        self.winset2 = set2

    def UpdateGame(self, game1, game2):
        self.wingame1 = game1
        self. wingame2 = game2

if __name__ == "__main__" :
    test = MomentumCaculater(0.5, 0.5, 0, 0.33)
    test.UpdateSet(2, 0)
    test.UpdateGame(0, 0)

    print(test.GetLeverage(0, 0, 1))