# -*- coding: utf-8 -*-

from z3 import Solver,Int
import math

class NumberPlace:

    def __init__(self, size):
        self.size = size
        self.solver = Solver()
        # introduce variables
        self.val = [[Int("val[%d,%d]" % (i,j)) for j in range(size)] for i in range(size)]


    #  1<=val<=9
    def _betweenOneToNine(self):
        for i in range(self.size):
            for j in range(self.size):
                self.solver.add(1 <= self.val[i][j], self.val[i][j] <= self.size)

    # row
    def _distinctRow(self):
        for i in range(self.size):
            tmpList = []
            for j in range(self.size):
                tmpList.append(self.val[i][j])
            self.solver.add(Distinct(tmpList))

    # column
    def _distinctColumn(self):
        for i in range(self.size):
            tmpList = []
            for j in range(self.size):
                tmpList.append(self.val[j][i])
            self.solver.add(Distinct(tmpList))

    # block
    def _distinctBlock(self, blockSize):
        for k in range(blockSize):
            for l in range(blockSize):
                tmpList = []
                for i in range(blockSize):
                    for j in range(blockSize):
                        tmpList.append(self.val[blockSize*k+i][blockSize*l+j])
                self.solver.add(Distinct(tmpList))

    def _setNum(self, x, y, num):
        self.solver.add(self.val[x-1][y-1]==num)


    # initialize problem
    def _setProb(self):
        i=0
        for p in self.problem:
            i += 1
            j = 0
            for pe in p:
                j += 1
                if pe != 0:
                    self._setNum(i, j, pe)
#                    print("("+str(i)+","+str(j)+") "+str(pe)) # for DEBUG


    def setRules(self, problem):
        self.problem = problem

        # set rules
        self._betweenOneToNine()
        self._distinctRow()
        self._distinctColumn()
        self._distinctBlock(int(math.sqrt(self.size)))

        # set problem
        self._setProb()

    def solve(self):
        r = self.solver.check()
        if r == sat:
            self.model = self.solver.model()
            return True
        else:
            return False

    def printOut(self):
        for i in range(self.size):
            for j in range(self.size):
                print("%d " % self.model[ self.val[i][j] ].as_long(), end="")
            print()

    def getCell(self, i, j):
        return self.model[ self.val[i][j] ].as_long()

    def getModel(self):
        return self.model


if __name__ == '__main__':
    # http://www.sudokugame.org/archive/printsudoku.php?nd=2&xh=1
    problem = [[0,0,0, 0,0,0, 0,0,0],
           [0,0,0, 0,0,1, 0,8,0],
           [6,4,0, 0,0,0, 7,0,0],
           [0,0,0, 0,0,3, 0,0,0],
           [0,0,1, 8,0,5, 0,0,0],
           [9,0,0, 0,0,0, 4,0,2],
           [0,0,0, 0,0,9, 3,5,0],
           [7,0,0, 0,6,0, 0,0,0],
           [0,0,0, 0,2,0, 0,0,0]]

    numplace = NumberPlace(9)
    numplace.setRules(problem)
    if numplace.solve():
        numplace.printOut()
        print(numplace.getModel())