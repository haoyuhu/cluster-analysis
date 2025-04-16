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
    __slots__ = ["x", "y", "group"]

    def __init__(self, x=0, y=0, group=0):
        self.x, self.y, self.group = x, y, group


def generatePoints(pointsNumber, radius):
    points = [Point() for _ in range(2 * pointsNumber)]
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
    for point in points:
        point.group = getNearestCenter(point, clusterCenterGroup)[0]
    return


def kMeans(points, clusterCenterNumber):
    clusterCenterGroup = [Point() for _ in range(clusterCenterNumber)]
    kMeansPlusPlus(points, clusterCenterGroup)
    clusterCenterTrace = [[clusterCenter] for clusterCenter in clusterCenterGroup]
    tolerableError, currentError = 5.0, FLOAT_MAX
    count = 0
    while currentError >= tolerableError:
        count += 1
        countCenterNumber = [0 for _ in range(clusterCenterNumber)]
        currentCenterGroup = [Point() for _ in range(clusterCenterNumber)]
        for point in points:
            currentCenterGroup[point.group].x += point.x
            currentCenterGroup[point.group].y += point.y
            countCenterNumber[point.group] += 1
        for index, center in enumerate(currentCenterGroup):
            center.x /= countCenterNumber[index]
            center.y /= countCenterNumber[index]
        currentError = 0.0
        for index, singleTrace in enumerate(clusterCenterTrace):
            singleTrace.append(currentCenterGroup[index])
            currentError += solveDistanceBetweenPoints(singleTrace[-1], singleTrace[-2])
            clusterCenterGroup[index] = copy.copy(currentCenterGroup[index])
        for point in points:
            point.group = getNearestCenter(point, clusterCenterGroup)[0]
    return clusterCenterGroup, clusterCenterTrace


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
    points = generatePoints(pointsNumber, radius)
    _, clusterCenterTrace = kMeans(points, clusterCenterNumber)
    showClusterAnalysisResults(points, clusterCenterTrace)


main()
