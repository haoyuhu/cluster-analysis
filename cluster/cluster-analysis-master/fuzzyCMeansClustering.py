import copy
import math
import random
import pylab

try:
    import psyco

    psyco.full()
except ImportError:
    pass

FLOAT_MAX = 1e100


class Point:
    __slots__ = ["x", "y", "group", "membership"]

    def __init__(self, clusterCenterNumber, x=0, y=0, group=0):
        self.x, self.y, self.group = x, y, group
        self.membership = [0.0 for _ in range(clusterCenterNumber)]


def generatePoints(pointsNumber, radius, clusterCenterNumber):
    points = [Point(clusterCenterNumber) for _ in range(2 * pointsNumber)]
    count = 0
    for point in points:
        count += 1
        r = random.random() * radius
        angle = random.random() * 2 * math.pi
        point.x = r * math.cos(angle)
        point.y = r * math.sin(angle)
        if count == pointsNumber - 1:
            break
    for index in range(pointsNumber, 2 * pointsNumber):
        points[index].x = 2 * radius * random.random() - radius
        points[index].y = 2 * radius * random.random() - radius
    return points


def solveDistanceBetweenPoints(pointA, pointB):
    return (pointA.x - pointB.x) * (pointA.x - pointB.x) + (pointA.y - pointB.y) * (pointA.y - pointB.y)


def getNearestCenter(point, clusterCenterGroup):
    minIndex = point.group
    minDistance = FLOAT_MAX
    for index, center in enumerate(clusterCenterGroup):
        distance = solveDistanceBetweenPoints(point, center)
        if (distance < minDistance):
            minDistance = distance
            minIndex = index
    return (minIndex, minDistance)


def kMeansPlusPlus(points, clusterCenterGroup):
    clusterCenterGroup[0] = copy.copy(random.choice(points))
    distanceGroup = [0.0 for _ in range(len(points))]
    sum = 0.0
    for index in range(1, len(clusterCenterGroup)):
        for i, point in enumerate(points):
            distanceGroup[i] = getNearestCenter(point, clusterCenterGroup[:index])[1]
            sum += distanceGroup[i]
        sum *= random.random()
        for i, distance in enumerate(distanceGroup):
            sum -= distance;
            if sum < 0:
                clusterCenterGroup[index] = copy.copy(points[i])
                break
    return


def fuzzyCMeansClustering(points, clusterCenterNumber, weight):
    clusterCenterGroup = [Point(clusterCenterNumber) for _ in range(clusterCenterNumber)]
    kMeansPlusPlus(points, clusterCenterGroup)
    clusterCenterTrace = [[clusterCenter] for clusterCenter in clusterCenterGroup]
    tolerableError, currentError = 1.0, FLOAT_MAX
    while currentError >= tolerableError:
        for point in points:
            getSingleMembership(point, clusterCenterGroup, weight)
        currentCenterGroup = [Point(clusterCenterNumber) for _ in range(clusterCenterNumber)]
        for centerIndex, center in enumerate(currentCenterGroup):
            upperSumX, upperSumY, lowerSum = 0.0, 0.0, 0.0
            for point in points:
                membershipWeight = pow(point.membership[centerIndex], weight)
                upperSumX += point.x * membershipWeight
                upperSumY += point.y * membershipWeight
                lowerSum += membershipWeight
            center.x = upperSumX / lowerSum
            center.y = upperSumY / lowerSum
        # update cluster center trace
        currentError = 0.0
        for index, singleTrace in enumerate(clusterCenterTrace):
            singleTrace.append(currentCenterGroup[index])
            currentError += solveDistanceBetweenPoints(singleTrace[-1], singleTrace[-2])
            clusterCenterGroup[index] = copy.copy(currentCenterGroup[index])
    for point in points:
        maxIndex, maxMembership = 0, 0.0
        for index, singleMembership in enumerate(point.membership):
            if singleMembership > maxMembership:
                maxMembership = singleMembership
                maxIndex = index
        point.group = maxIndex
    return clusterCenterGroup, clusterCenterTrace


def getSingleMembership(point, clusterCenterGroup, weight):
    distanceFromPoint2ClusterCenterGroup = [solveDistanceBetweenPoints(point, clusterCenterGroup[index]) for index in
                                            range(len(clusterCenterGroup))]
    for centerIndex, singleMembership in enumerate(point.membership):
        sum = 0.0
        isCoincide = [False, 0]
        for index, distance in enumerate(distanceFromPoint2ClusterCenterGroup):
            if distance == 0:
                isCoincide[0] = True
                isCoincide[1] = index
                break
            sum += pow(float(distanceFromPoint2ClusterCenterGroup[centerIndex] / distance), 1.0 / (weight - 1.0))
        if isCoincide[0]:
            if isCoincide[1] == centerIndex:
                point.membership[centerIndex] = 1.0
            else:
                point.membership[centerIndex] = 0.0
        else:
            point.membership[centerIndex] = 1.0 / sum


def showClusterAnalysisResults(points, clusterCenterTrace):
    colorStore = ['or', 'og', 'ob', 'oc', 'om', 'oy', 'ok']
    pylab.figure(figsize=(9, 9), dpi=80)
    for point in points:
        color = ''
        if point.group >= len(colorStore):
            color = colorStore[-1]
        else:
            color = colorStore[point.group]
        pylab.plot(point.x, point.y, color)
    for singleTrace in clusterCenterTrace:
        pylab.plot([center.x for center in singleTrace], [center.y for center in singleTrace], 'k')
    pylab.show()


def main():
    clusterCenterNumber = 5
    pointsNumber = 2000
    radius = 10
    weight = 2
    points = generatePoints(pointsNumber, radius, clusterCenterNumber)
    _, clusterCenterTrace = fuzzyCMeansClustering(points, clusterCenterNumber, weight)
    showClusterAnalysisResults(points, clusterCenterTrace)


main()
