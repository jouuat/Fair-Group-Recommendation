import copy
import time


# Build the solution
class Solution:

    def __init__(self, PriceToEarningsRatio, PriceToBookRatio,
                 PriceToEarningsToGrowth, ReturnOnEquity, DebtToEquityRatio, LiquidityRatio, TotalPuntuation, numRatios):
        self.PriceToEarningsRatio = PriceToEarningsRatio
        self.PriceToBookRatio = PriceToBookRatio
        self.PriceToEarningsToGrowth = PriceToEarningsToGrowth
        self.ReturnOnEquity = ReturnOnEquity
        self.DebtToEquityRatio = DebtToEquityRatio
        self.LiquidityRatio = LiquidityRatio
        self.TotalPuntuation = TotalPuntuation
        self.numRatios = numRatios


# -----------------------------save the solution---------------------------------------

    def str(self, PriceToEarningsRatio, PriceToBookRatio, PriceToEarningsToGrowth, ReturnOnEquity, DebtToEquityRatio, LiquidityRatio):

        strSolution = 'PriceToEarningsRatio = %s;\n' % str(PriceToEarningsRatio)
        strSolution += 'PriceToBookRatio = %s;\n' % str(PriceToBookRatio)
        strSolution += 'PriceToEarningsToGrowth = %s;\n' % str(PriceToEarningsToGrowth)
        strSolution += 'ReturnOnEquity = %s;\n' % str(ReturnOnEquity)
        strSolution += 'DebtToEquityRatio = %s;\n' % str(DebtToEquityRatio)
        strSolution += 'LiquidityRatio = %s;\n' % str(LiquidityRatio)

        return(strSolution)

    def saveToFile(self, filePath, PriceToEarningsRatio, PriceToBookRatio, PriceToEarningsToGrowth, ReturnOnEquity, DebtToEquityRatio, LiquidityRatio):
        f = open(filePath, 'w')
        f.write(self.str(PriceToEarningsRatio, PriceToBookRatio, PriceToEarningsToGrowth,
                         ReturnOnEquity, DebtToEquityRatio, LiquidityRatio))
        f.close()
