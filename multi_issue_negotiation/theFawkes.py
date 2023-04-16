from ast import If
from agent import Agent
from utils import get_utility, floatToBits
import random
import math
import copy
import numpy
from functools import reduce

class BidDetails:
    def __init__(self, bid, myUndiscountedUtil, time):
        self.serialVersionUID = -6111983303592311613
        self.bid = bid
        self.myUndiscountedUtil = myUndiscountedUtil
        self.time = time
        
    def getBid(self):
        return self.bid
    
    def setBid(self, bid):
        self.bid = bid

    def getMyUndiscountedUtil(self):
        return self.myUndiscountedUtil
    
    def setMyUndiscountedUtil(self, utility):
        self.myUndiscountedUtil = utility

    def getTime(self):
        return self.time

    def setTime(self, time):
        self.time = time

    def _hashCode(self):
        if self.bid is None:
            return 0
        hashValue = 0
        for value in self.bid:
            value_len = len(value)
            h = 0
            for i in range(value_len):
                h = int(31 * h + ord(value[i]))
            hashValue = int(hashValue + h)
        return hashValue

    # public int compareTo(BidDetails other)

    def __hash__(self):
        prime = 31
        result = 1
        result = prime * result + self._hashCode()
        temp = floatToBits(self.myUndiscountedUtil)
        result = prime * result + int(temp ^ (temp >> 32))
        temp = floatToBits(self.time)
        result = prime * result + int(temp ^ (temp >> 32))
        return result

    def __eq__(self, other):
        if self == other:
            return True
        if other is None:
            return False
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        if self.bid is None:
            if other.bid is not None:
                return False
        elif self.bid != other.bid:
            return False
        if floatToBits(self.myUndiscountedUtil) != floatToBits(other.myUndiscountedUtil):
            return False
        if floatToBits(self.time) != floatToBits(other.time):
            return False
        return True

    def __lt__(self, other):
        otherUtil = other.getMyUndiscountedUtil()
        if self.myUndiscountedUtil > otherUtil:
            return True
        else:
            return False


class JWave_Daubechie:
    def __init__(self):
        self._waveLength = 8
        self._scales = [0] * self._waveLength
        self._scales[0] = 0.32580343
        self._scales[1] = 1.01094572
        self._scales[2] = 0.8922014
        self._scales[3] = -0.03967503
        self._scales[4] = -0.2645071
        self._scales[5] = 0.0436163
        self._scales[6] = 0.0465036
        self._scales[7] = -0.01498699
        #  normalize to square root of 2 for being orthonormal
        self.sqrt02 = 1.4142135623730951
        for i in range(self._waveLength):
            self._scales[i] /= self.sqrt02
        self._coeffs = [0] * self._waveLength
        self._coeffs[0] = self._scales[7]
        self._coeffs[1] = - self._scales[6]
        self._coeffs[2] = self._scales[5]
        self._coeffs[3] = - self._scales[4]
        self._coeffs[4] = self._scales[3]
        self._coeffs[5] = - self._scales[2]
        self._coeffs[6] = self._scales[1]
        self._coeffs[7] = - self._scales[0]

    def getWaveLength(self):
        return self._waveLength

    def forward(self, arrTime):
        arrHilb = [0] * len(arrTime)
        k = len(arrTime) >> 1
        h = len(arrTime) >> 1
        for i in range(h):
            for j in range(self._waveLength):
                k = ( i << 1 ) + j
                while k >= len(arrTime):
                    k -= len(arrTime)
                arrHilb[i] += arrTime[k] * self._scales[j]
                arrHilb[i + h] += arrTime[k] * self._coeffs[j]
        return arrHilb


class JWave_DiscreteWaveletTransform:
    def __init__(self, wavelet, iteration=-1):
        self._iteration = iteration
        self._wavelet = wavelet
    
    def forwardWavelet(self, arrTime):
        # arrHilb = [0.0] * len(arrTime)
        arrHilb = copy.deepcopy(arrTime)
        h = len(arrTime)
        minWaveLength = self._wavelet.getWaveLength()
        if h >= minWaveLength:
            while h >= minWaveLength:
                iBuf = [0] * h
                for i in range(h):
                    iBuf[i] = arrHilb[i]
                oBuf = self._wavelet.forward(iBuf)
                for i in range(h):
                    arrHilb[i] = oBuf[i]
                h = h >> 1
        return arrHilb

class SmoothingPolynomial:
    def __init__(self, coeff) -> None:
        self.coeff = copy.deepcopy(coeff)
        
    def evaluate(self, x):
        res = self.coeff[len(self.coeff) - 1]
        for i in range(len(self.coeff)-2, -1, -1):
            res = self.coeff[i] + x * res
        return res

    def derivative(self, x, n=1):
        if n < 0:
            raise Exception("n < 0")
        elif n == 0:
            return self.evaluate(x)
        elif n >= len(self.coeff):
            return 0
        else:
            res = self.getCoeffDer(len(self.coeff) - 1, n)
            for i in range(len(self.coeff)-2, n-1, -1):
                res = self.getCoeffDer(i, n) + (x * res)
            return res

    def getCoefficient(self, i):
        return self.coeff[i]

    def getCoeffDer(self, i, n):
        coeffDer = self.coeff[i]
        for j in range(i, i-n, -1):
            coeffDer *= j
        return coeffDer

class SmoothingCubicSpline:
    def __init__(self, x, y, rho,  w=None) -> None:  
        if len(x) != len(y):
            raise Exception('x.length != y.length')
        elif w is not None and len(x) != len(w):
            raise Exception('x.length != w.length')
        elif rho < 0 or rho > 1:
            raise Exception('rho not in [0, 1]')
        else:
            self.splineVector = [None] * (len(x) + 1)
            self.x = copy.deepcopy(x) 
            self.y = copy.deepcopy(y)
            self.weight = [0.0] * len(x)
            self.rho = rho
            if w is None:
                for i in range(len(self.weight)):
                    self.weight[i] = 1
            else:
                for i in range(len(self.weight)):
                    self.weight[i] = w[i]
            self.resolve()

    def resolve(self):
        n = len(self.x)
        h = [0.0] * n
        r = [0.0] * n
        u = [0.0] * n
        v = [0.0] * n
        w = [0.0] * n
        q = [0.0] * (n+1)
        sigma = [0.0] * len(self.weight)
        for i in range(len(sigma)):
            if self.weight[i] <= 0:
                sigma[i] = 1.0e100
            else:
                sigma[i] = 1 / math.sqrt(self.weight[i])
        n = len(self.x) - 1
        if self.rho <= 0:
            mu = 1.0e100
        else:
            mu = ( 2 * ( 1 - self.rho ) ) / ( 3 * self.rho )
        h[0] = self.x[1] - self.x[0]
        r[0] = 3 / h[0]
        for i in range(1, n):
            h[i] = self.x[i + 1] - self.x[i]
            r[i] = 3 / h[i]
            q[i] = ( 3 * ( self.y[i + 1] - self.y[i] ) / h[i] ) - ( 3 * ( self.y[i] - self.y[i - 1] ) / h[i - 1] )
        for i in range(1, n):
            u[i] = ( r[i - 1] * r[i - 1] * sigma[i - 1] ) + ( ( r[i - 1] + r[i] ) * ( r[i - 1] + r[i] ) * sigma[i] ) + ( r[i] * r[i] * sigma[i + 1] )
            u[i] = mu * u[i] + 2 * ( self.x[i + 1] - self.x[i - 1] )
            v[i] = ( -( r[i - 1] + r[i] ) * r[i] * sigma[i] ) - ( r[i] * ( r[i] + r[i + 1] ) * sigma[i + 1] )
            v[i] = ( mu * v[i] ) + h[i]
            w[i] = mu * r[i] * r[i + 1] * sigma[i + 1]
        q = self.Quincunx( u, v, w, q )
        # extrapolation a gauche
        params = [0] * 4
        params[0] = self.y[0] - ( mu * r[0] * q[1] * sigma[0] )
        dd1 = self.y[1] - ( mu * ( ( -r[0] - r[1] ) * q[1] + r[1] * q[2] ) * sigma[1] )
        params[1] = ( dd1 - params[0] ) / h[0] - ( q[1] * h[0] / 3 )
        self.splineVector[0] = SmoothingPolynomial( params )
        # premier polynome
        params[0] = self.y[0] - ( mu * r[0] * q[1] * sigma[0] )
        dd2 = self.y[1] - ( mu * ( ( -r[0] - r[1] ) * q[1] + r[1] * q[2] ) * sigma[1] )
        params[3] = q[1] / ( 3 * h[0] )
        params[2] = 0
        params[1] = ( ( dd2 - params[0] ) / h[0] ) - ( q[1] * h[0] / 3 )
        self.splineVector[1] = SmoothingPolynomial( params )
        # les polynomes suivants
        j = 0
        for j in range(1, n):
            params[3] = ( q[j + 1] - q[j] ) / ( 3 * h[j] )
            params[2] = q[j]
            params[1] = ( ( q[j] + q[j - 1] ) * h[j - 1] ) + self.splineVector[j].getCoefficient( 1 )
            params[0] = ( r[j - 1] * q[j - 1] ) + ( ( -r[j - 1] - r[j] ) * q[j] ) + ( r[j] * q[j + 1] )
            params[0] = self.y[j] - ( mu * params[0] * sigma[j] )
            self.splineVector[j + 1] = SmoothingPolynomial( params )     
        # extrapolation a droite
        j = n
        params[3] = 0
        params[2] = 0
        params[1] = self.splineVector[j].derivative( self.x[n] - self.x[n - 1] )
        params[0] = self.splineVector[j].evaluate( self.x[n] - self.x[n - 1] )
        self.splineVector[n + 1] = SmoothingPolynomial( params )

    def evaluate(self, z):
        i = self.getFitPolynomialIndex(z)
        if i == 0:
            returned = self.splineVector[i].evaluate( z - self.x[0] )
        else:
            returned = self.splineVector[i].evaluate( z - self.x[i - 1] )
        if math.isnan(returned) or returned < 0:
            returned = 0
        elif math.isinf(returned) or returned > 1:
            returned = 1
        return returned

    def Quincunx(self, u, v, w, q):
        u[0] = 0
        v[1] /= u[1]
        w[1] /= u[1]
        for j in range(2, len(u)-1):
            u[j] = u[j] - ( u[j - 2] * w[j - 2] * w[j - 2] ) - ( u[j - 1] * v[j - 1] * v[j - 1] )
            v[j] = ( v[j] - ( u[j - 1] * v[j - 1] * w[j - 1] ) ) / u[j]
            w[j] /= u[j]
        q[1] -= v[0] * q[0]
        for j in range(2, len(u)-1):
            q[j] = q[j] - ( v[j - 1] * q[j - 1] ) - ( w[j - 2] * q[j - 2] )
        for j in range(1, len(u)-1):
            q[j] /= u[j]
        q[len(u)-1] = 0
        for j in range(len(u)-3, 0, -1):
            q[j] = q[j] - ( v[j] * q[j + 1] ) - ( w[j] * q[j + 2] )
        return q

    def getFitPolynomialIndex(self, x):
        j = len(self.x) - 1
        if x > self.x[j]:
            return  j+1
        tmp = 0
        i = 0
        while (i+1) != j:
            if x > self.x[tmp]:
                i = tmp
                tmp = i + ( ( j - i ) / 2 )
            else:
                j = tmp
                tmp = i + ( ( j - i ) / 2 )
            if j == 0:
                i -= 1
        return i+1

class TheFawkes_OM:
    def __init__(self):
        self.timeIndexFactor = 0.0
        self.chi = []
        self.ratio = []
        self.alpha = None
        self.learnCoef = 0.0
        self.learnValueAddition = 0
        self.amountOfIssues = 0
        self.maxOpponentBidTimeDiff = 0.0
        self.oppo_prefer = []
        self.oppo_issue_value = []
        self.oppo_issue_value_non_normalized = []
        self.oppo_issue_num = []
        self.cleared = False
        # self.OpponentBidHistory = []
    
    def init(self, agent):
        self.agent = agent
        self.oppo_issue_num = agent.issue_num
        self.oppo_issue_value = copy.deepcopy(agent.issue_value)
        self.oppo_prefer = copy.deepcopy(agent.prefer)
        self.oppo_issue_value_non_normalized = copy.deepcopy(agent.issue_value)
        # self.OpponentBidHistory = self.agent.OpponentBidHistory
        self.cleared = False
        self.timeIndexFactor = agent.max_round
        self.learnCoef = 0.8
        self.learnValueAddition = 1
        self.initializeFrequencyModel()
    
    def initializeFrequencyModel(self):
        self.amountOfIssues = self.oppo_issue_num
        commonWeight = 1 / self.amountOfIssues
        self.oppo_prefer = [commonWeight] * self.amountOfIssues
        for i in range(self.amountOfIssues):
            for value in self.oppo_issue_value[i].keys():
                self.oppo_issue_value_non_normalized[i][value] = 1

    def determineDifference(self, firstBid, secondBid):
        diff = {}
        for i in range(self.amountOfIssues):
            firstValue = firstBid[i]
            secondValue = secondBid[i]
            if firstValue == secondValue:
                diff[i] = 0
            else:
                diff[i] = 1
        return diff

    def normalizeIssueValue(self):
       
        for i in range(self.oppo_issue_num):
            maxValue = 0.0
            for key in self.oppo_issue_value_non_normalized[i].keys():                    
                if self.oppo_issue_value_non_normalized[i][key] > maxValue:
                    maxValue = self.oppo_issue_value_non_normalized[i][key]
            for key in self.oppo_issue_value[i].keys():
                self.oppo_issue_value[i][key] = self.oppo_issue_value_non_normalized[i][key] / maxValue


    def getBidEvaluation(self, bid):
        self.normalizeIssueValue()
        return get_utility(bid, self.oppo_prefer, 1, 'DISCRETE', self.oppo_issue_value)
    
    def getMaxOpponentBidTimeDiff(self):
        return min(0.1, self.maxOpponentBidTimeDiff)
    
    def updateModel(self, bid, time):
        opponentHistory = []
        len_ = len(self.agent.OpponentBidHistory)
        # self.agent.OpponentBidHistory.reverse()
        # print("self.agent.OpponentBidHistory:", self.agent.OpponentBidHistory)
        for i in range(len_):
            opponentHistory.append(self.agent.OpponentBidHistory[len_-1-i])
        # print("opponentHistory:", opponentHistory)
        # exit(-1)
        if len(opponentHistory) < 2:
            return
        previousBid = opponentHistory[1].getBid()
        lastDiffSet = self.determineDifference(bid, previousBid)
        numberOfUnchanged = 0
        for key in lastDiffSet.keys():
            if lastDiffSet[key] == 0:
                numberOfUnchanged += 1
        
        goldenValue = self.learnCoef / self.amountOfIssues
        totalSum = 1 + (goldenValue * numberOfUnchanged)
        maximumWeight = 1 - ((self.amountOfIssues * goldenValue) / totalSum)
        for i in lastDiffSet.keys():
            if lastDiffSet[i] == 0 and self.oppo_prefer[i] < maximumWeight:
                self.oppo_prefer[i] = (self.oppo_prefer[i] + goldenValue) / totalSum
            else:
                self.oppo_prefer[i] = self.oppo_prefer[i] / totalSum
        for i in range(self.amountOfIssues):
            vd = bid[i]
            self.oppo_issue_value_non_normalized[i][vd] += self.learnValueAddition

        map_ = {}
        maxtime = -1
        prevBidTime = self.agent.OpponentBidHistory[-1].getTime()
        for biddetails in self.agent.OpponentBidHistory:
            diff = prevBidTime - biddetails.getTime()
            if diff > self.maxOpponentBidTimeDiff:
                self.maxOpponentBidTimeDiff = diff
            prevBidTime = biddetails.getTime()
            bidtime = math.floor(biddetails.getTime() * self.timeIndexFactor)
            map_[bidtime] = biddetails.getMyUndiscountedUtil()
            if bidtime > maxtime:
                maxtime = bidtime
        
        self.chi = [0.0] * (maxtime+1)
        for key in map_.keys():
            self.chi[key] = map_[key]
        
        dwt = JWave_DiscreteWaveletTransform(JWave_Daubechie())
        decomposition = dwt.forwardWavelet(self.chi)
        N = len(decomposition)
        x = [0] * (N+1)
        y = [0] * (N+1)
        for i in range(N):
            x[i] = i
            y[i] = decomposition[i]

        self.alpha = SmoothingCubicSpline(x, y, 1e-16)
        self.ratio = [0.0] * (maxtime + 1)
        for i in range(maxtime+1):
            if self.chi[i] == 0:
                currentRatio = 0
            else:
                currentRatio = self.alpha.evaluate(i) / self.chi[i]
            if math.isinf(currentRatio):
                currentRatio = 0
            elif math.isnan(currentRatio):
                currentRatio = 1
            self.ratio[i] = currentRatio

class TheFawkes_OMS:
    def __init__(self) -> None:
        self.lastTen = []
        self.secondBestCounter = 1
        self.model = None
        self.RANGE_INCREMENT = 0.01
        self.EXPECTED_BIDS_IN_WINDOW = 100
        self.INITIAL_WINDOW_RANGE = 0.01
    
    def init(self, prefer, issue_num, issue_value, oppModel):
        self.initializeAgent(prefer, issue_num, issue_value, oppModel)
    
    def initializeAgent(self, prefer, issue_num, issue_value, oppModel):
        self.prefer = prefer
        self.issue_num = issue_num
        self.issue_value = copy.deepcopy(issue_value)
        self.model = oppModel
        self.lastTen = [0] * 11

    def mySort(self, biddetail):
        return self.model.getBidEvaluation(biddetail.getBid())
    
    # return：BidDetails
    # param：List<BidDetails>
    def getBid(self, list_):
        list_.sort(key=self.mySort, reverse=True)
        opponentBestBid = list_[0]
        allEqual = True
        for bid in self.lastTen:
            if bid != opponentBestBid.getMyUndiscountedUtil():
                allEqual = False
        if allEqual:
            self.secondBestCounter += 1
            if len(list_) > 1:
                opponentBestBid = list_[1]
        self.lastTen.append(opponentBestBid.getMyUndiscountedUtil())
        if len(self.lastTen) > 10:
            self.lastTen.remove(self.lastTen[0])
        return opponentBestBid
    
    def getSecondBestCount(self):
        return self.secondBestCounter
    
    def canUpdateOM(self):
        return True
    
class Fawkes_Offering:
    def __init__(self) -> None:
        self.nextBid = None
        self.opponentModel = None
        self.omStrategy = None
        self.endNegotiation = False
        self.beta = 0
        self.rho = 0
        self.nu = 0
        self.discountFactor = 0
        self.OM = None
        self.OMS = None
        self.DOUBLE_MAX = 1.7976931348623158E+308     # 64 bit floating point max value
        self.DOUBLE_MIN = 2.2250738585072014E-308     # 64 bit floating point min value

    def init(self, agent, oppModel, omStrategy):
        self.agent = agent
        self.opponentModel = oppModel
        self.omStrategy = omStrategy
        self.endNegotiation = False
        self.OM = oppModel
        self.OMS = omStrategy
        # this.negotiationSession.setOutcomeSpace(new SortedOutcomeSpace(this.negotiationSession.getUtilitySpace()));
        self.beta = 0.002
        self.rho = 0.8
        self.nu = 0.2
        self.discountFactor = self.agent.discount
        if self.discountFactor == 0:
            self.discountFactor = 1

    def isEndNegotiation(self):
        return self.endNegotiation

    def setNextBid(self, nextBid):
        self.nextBid = nextBid
        
    def determineOpeningBid(self):
        ret = BidDetails(self.agent.bestBid, self.agent.getUtility(self.agent.bestBid), self.agent.relative_t)
        return ret
    
    def determineNextBid(self):
        if len(self.agent.OpponentBidHistory) > 1:
            returned = self.responseMechanism(self.getTargetUtility())
        else:
            returned = self.determineOpeningBid()
        returned.setTime(self.agent.relative_t)
        return returned

    def getBidsinRange(self, lowerbound, upperbound):
        bidsInRange = []
        if self.agent.allBids is None:
            self.agent.allBids = self.agent.getAllBids()
        for _bid in self.agent.allBids:
            utility = get_utility(_bid, self.agent.prefer, 1, self.agent.domain_type, self.agent.issue_value)
            if utility > lowerbound and utility < upperbound:
                bidsInRange.append(BidDetails(_bid, utility, 0))

        return bidsInRange
    
    def responseMechanism(self, targetUtility):
        timeCorrection = 1 - (self.agent.relative_t / 10.0)
        inertiaCorrection = self.OMS.getSecondBestCount() / (len(self.agent.OwnBidHistory) * 10)
        lowerBound = (timeCorrection - inertiaCorrection) * targetUtility
        possibleBids = self.getBidsinRange(lowerBound, 2)
        if len(possibleBids) == 0:
            return self.agent.OwnBidHistory[-1]
        else:
            return self.omStrategy.getBid(possibleBids)

    def getTargetUtility(self):
        currentTime = self.agent.relative_t
        maxDiff = self.DOUBLE_MIN
        minDiff = self.DOUBLE_MAX
        optimisticEstimatedTime = 0
        optimisticEstimatedUtility = 0
        pessimisticEstimatedTime = 0
        pessimisticEstimatedUtility = 0
        for time in numpy.arange(currentTime, min(currentTime+self.OM.getMaxOpponentBidTimeDiff()*10, 1), self.OM.getMaxOpponentBidTimeDiff()):
            reservedUtility = self.reservedUtility(time)
            discountedReservedUtility = self.discountedUtility(reservedUtility, time)
            estimatedReceivedUtility = self.estimatedReceivedUtility(time)
            discountedEstimatedReceivedUtility = self.discountedUtility(estimatedReceivedUtility, time)
            diff = discountedEstimatedReceivedUtility - discountedReservedUtility
            if discountedEstimatedReceivedUtility >= discountedReservedUtility and diff > maxDiff:
                maxDiff = diff
                optimisticEstimatedTime = time
                optimisticEstimatedUtility = estimatedReceivedUtility
            if maxDiff == self.DOUBLE_MIN:
                absoluteDiff = abs(diff)
                if absoluteDiff < minDiff:
                    minDiff = absoluteDiff
                    pessimisticEstimatedTime = time
                    pessimisticEstimatedUtility = estimatedReceivedUtility
        if maxDiff == self.DOUBLE_MIN:
            xsi = (math.pow(self.rho, -1) * self.discountedUtility(pessimisticEstimatedUtility, pessimisticEstimatedTime)) / self.discountedUtility(self.reservedUtility(pessimisticEstimatedTime), pessimisticEstimatedTime)
            if xsi > 1:
                estimatedTime = pessimisticEstimatedTime
                estimatedUtility = pessimisticEstimatedUtility
            else:
                estimatedTime = -1
                estimatedUtility = -1
        else:
            estimatedTime = optimisticEstimatedTime
            estimatedUtility = optimisticEstimatedUtility
        if estimatedUtility == -1:
            targetUtility = self.reservedUtility(currentTime)
        else:
            myPreviousBid = self.agent.OwnBidHistory[-1]
            myPreviousUtil = myPreviousBid.getMyUndiscountedUtil()
            factor = (currentTime - estimatedTime) / (myPreviousBid.getTime() - estimatedTime)
            targetUtility = estimatedUtility + (myPreviousUtil - estimatedUtility) * factor
        return targetUtility
    
    def reservedUtility(self, time):
        reservationValue = self.agent.reservation
        myBestBidUtility = self.agent.getUtility(self.agent.bestBid)
        return reservationValue + (1 - math.pow(time, 1 / self.beta)) * ((myBestBidUtility * math.pow(self.discountFactor, self.nu)) - reservationValue)

    def estimatedReceivedUtility(self, time):
        timeIndex = int(math.floor(time * self.OM.timeIndexFactor))
        return self.OM.alpha.evaluate(timeIndex) * (1 + Fawkews_Math.getStandardDeviation(self.OM.ratio))

    def discountedUtility(self, utility, time):
        return utility * math.pow(self.discountFactor, time)

class Fawkews_Math:
    @staticmethod
    def getMean(values):
        mean = 0
        if values is not None and len(values) > 0:
            for value in values:
                mean += value
            mean /= len(values)
        return mean
    
    @staticmethod
    def getStandardDeviation(values):
        deviation = 0
        if values is not None and len(values) > 0:
            mean = Fawkews_Math.getMean(values)
            for value in values:
                delta = value - mean
                deviation += delta * delta
            deviation = math.sqrt( deviation / len(values) )
        return deviation

class AC_TheFawkes:
    def __init__(self) -> None:
        self.offeringStrategy = None
        self.opponentModel = None
        self.OM = None
        self.minimumAcceptable = 0

    def lists_combination(self, lists, code=''):
        
            
        def myfunc(list1, list2):
            return [str(i)+code+str(j) for i in list1 for j in list2]

        return reduce(myfunc, lists)

    def getAllOutcomes(self):
        _allBids = []
        issueValues = self.getIssueValues()
        resList = self.lists_combination(issueValues,'@')
        for i in range(len(resList)):
            tmpList = resList[i].split('@')
            _allBids.append(BidDetails(tmpList, get_utility(tmpList, self.agent.prefer, 1, 'DISCRETE', self.agent.issue_value), 0))
        return _allBids

    def getIssueValues(self):
        retvals = []          
        for i in range(self.agent.issue_num):
            retvals.append(list(self.agent.issue_value[i].keys()))
        return retvals  # list, element i means i_th issue's all possible values,like ["dell", "lenova", "HP"]

    def init(self, agent, biddingStrategy, oppModel):
        self.agent = agent
        self.offeringStrategy = biddingStrategy
        self.opponentModel = oppModel
        self.OM = oppModel
        allBids = self.getAllOutcomes()
        total = 0
        for bid in allBids:
            total += bid.getMyUndiscountedUtil()
        self.minimumAcceptable = total / len(allBids)

    def filterBetweenTime(self, bidDetaisList, t1, t2):
        return self.filterBetween(0, 1.1, t1, t2, bidDetaisList)
    
    def filterBetween(self, minU, maxU, minT, maxT, bidDetaisList):
        bestBidDetail = None
        maxUtil = -1
        for b in bidDetaisList:
            if b.getMyUndiscountedUtil() > minU and b.getMyUndiscountedUtil() <= maxU and b.getTime() > minT and b.getTime() <= maxT:
                if b.getMyUndiscountedUtil() > maxUtil:
                    maxUtil = b.getMyUndiscountedUtil()
                    bestBidDetail = b
        return bestBidDetail

    def determineAcceptability(self):
        lastOpponentBid = self.agent.OpponentBidHistory[-1]
        lastOpponentBidUtility = lastOpponentBid.getMyUndiscountedUtil()
        myNextBid = self.offeringStrategy.determineNextBid()
        myNextBidUtility = myNextBid.getMyUndiscountedUtil()
        if lastOpponentBidUtility < self.minimumAcceptable:
            return "Reject"
        elif lastOpponentBidUtility >= myNextBidUtility:
            return "Accept"
        elif self.agent.relative_t >= (1 - self.OM.getMaxOpponentBidTimeDiff()):
            time = self.agent.relative_t
            bestOpponentBid = self.filterBetweenTime(self.agent.OpponentBidHistory, time - (self.OM.getMaxOpponentBidTimeDiff() * 10), time)
            bestOpponentBidUtility = bestOpponentBid.getMyUndiscountedUtil()
            if lastOpponentBidUtility >= bestOpponentBidUtility:
                return "Accept"
            else:
                return "Reject"
        else:
            return "Reject"

class TheFawkes(Agent):
    def __init__(self, max_round, name="TheFawkes agent", u_max=1, u_min=0.1, issue_num=3):
        super().__init__(max_round=max_round, name=name, u_max=u_max, u_min=u_min, issue_num=issue_num)
        self.acceptConditions = None
        self.offeringStrategy = None
        self.opponentModel = None
        self.omStrategy = None
        self.oppBid = None
        self.OpponentBidHistory = [] # BidDetails
        self.OwnBidHistory = [] # BidDetails

    def reset(self):
        super().reset()
        self.acceptConditions = None
        self.offeringStrategy = None
        self.opponentModel = None
        self.omStrategy = None
        self.oppBid = None
        self.OpponentBidHistory = [] # BidDetails
        self.OwnBidHistory = [] # BidDetails
        self.agentSetup()

    def agentSetup(self):
        self.opponentModel = TheFawkes_OM()
        self.opponentModel.init(self)
        self.omStrategy = TheFawkes_OMS()
        self.omStrategy.init(self.prefer, self.issue_num, self.issue_value, self.opponentModel)
        self.offeringStrategy = Fawkes_Offering()
        self.acceptConditions = AC_TheFawkes()
        self.offeringStrategy.init(self, self.opponentModel, self.omStrategy)
        self.acceptConditions.init(self, self.offeringStrategy, self.opponentModel)
        
    def getUtility(self, bid):
        return get_utility(bid, self.prefer, 1, self.domain_type, self.issue_value)

    def receive(self, last_action=None):
        if last_action is not None:
            self.offer_received.append(last_action)
            last_util = self.getUtility(last_action)
            self.utility_received.append(last_util)
            opponentBid = BidDetails(last_action, last_util, self.relative_t)
            self.OpponentBidHistory.append(opponentBid)
            if self.opponentModel is not None:
                if self.omStrategy.canUpdateOM():
                    # print('self.OpponentBidHistory: ', self.OpponentBidHistory)
                    self.opponentModel.updateModel(last_action, self.relative_t)
            

    def generateRandomBid(self):
        return [random.choice(list(self.issue_value[i].keys())) for i in range(self.issue_num)]

    def gen_offer(self):
        bid = None
        if len(self.OwnBidHistory) == 0:
            bid = self.offeringStrategy.determineOpeningBid()
        else:
            bid = self.offeringStrategy.determineNextBid()
            if self.offeringStrategy.isEndNegotiation():
                self.terminate = True
                return None
        if bid is None:
            print("Error in code, null bid was given")
            self.accept = True
            return None
        else:
            self.offeringStrategy.setNextBid(bid)
        decision = "Reject"
        if len(self.OpponentBidHistory) != 0:
            decision = self.acceptConditions.determineAcceptability()
        if decision == "Break":
            self.terminate = True
            return None
        if decision == "Reject":
            self.OwnBidHistory.append(bid)
            self.offer_proposed.append(bid.getBid())
            self.utility_proposed.append(bid.getMyUndiscountedUtil())
            return bid.getBid()
        else:
            self.accept = True
            return None