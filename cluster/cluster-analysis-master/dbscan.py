import math
import random
import pylab

try:
    import psyco

    psyco.full()
except ImportError:
    pass

FLOAT_MAX = 1e100

CORE_POINT_TYPE = -2
BOUNDARY_POINT_TYPE = 1  # ALL NONE-NEGATIVE INTEGERS CAN BE BOUNDARY POINT TYPE
OTHER_POINT_TYPE = -1


class Point:
    __slots__ = ["x", "y", "group", "pointType"]

    def __init__(self, x=0, y=0, group=0, pointType=-1):
        self.x, self.y, self.group, self.pointType = x, y, group, pointType


def generatePoints(pointsNumber, radius):
    points = [Point() for _ in range(4 * pointsNumber)]
    originX = [-radius, -radius, radius, radius]
    originY = [-radius, radius, -radius, radius]
    count = 0
    countCenter = 0
    for index, point in enumerate(points):
        count += 1
        r = random.random() * radius
        angle = random.random() * 2 * math.pi
        point.x = r * math.cos(angle) + originX[countCenter]
        point.y = r * math.sin(angle) + originY[countCenter]
        point.group = index
        if count >= pointsNumber * (countCenter + 1):
            countCenter += 1
    return points


def solveDistanceBetweenPoints(pointA, pointB):
    return (pointA.x - pointB.x) * (pointA.x - pointB.x) + (pointA.y - pointB.y) * (pointA.y - pointB.y)


def isInPointBoundary(centerPoint, customPoint, halfScale):
    return customPoint.x <= centerPoint.x + halfScale and customPoint.x >= centerPoint.x - halfScale and customPoint.y <= centerPoint.y + halfScale and customPoint.y >= centerPoint.y - halfScale


def getPointsNumberWithinBoundary(points, halfScale):
    pointsIndexGroupWithinBoundary = [[] for _ in range(len(points))]
    for centerIndex, centerPoint in enumerate(points):
        for index, customPoint in enumerate(points):
            if centerIndex != index and isInPointBoundary(centerPoint, customPoint, halfScale):
                pointsIndexGroupWithinBoundary[centerIndex].append(index)
    return pointsIndexGroupWithinBoundary


def decidePointsType(points, pointsIndexGroupWithinBoundary, minPointsNumber):
    for index, customPointsGroup in enumerate(pointsIndexGroupWithinBoundary):
        if len(customPointsGroup) >= minPointsNumber:
            points[index].pointType = CORE_POINT_TYPE
    for index, customPointsGroup in enumerate(pointsIndexGroupWithinBoundary):
        if len(customPointsGroup) < minPointsNumber:
            for customPointIndex in customPointsGroup:
                if points[customPointIndex].pointType == CORE_POINT_TYPE:
                    points[index].pointType = customPointIndex


def mergeGroup(points, fromIndex, toIndex):
    for point in points:
        if point.group == fromIndex:
            point.group = toIndex


def dbscan(points, pointsIndexGroupWithinBoundary, clusterCenterNumber):
    countGroupsNumber = {index: 1 for index in range(len(points))}
    for index, point in enumerate(points):
        if point.pointType == CORE_POINT_TYPE:
            for customPointIndex in pointsIndexGroupWithinBoundary[index]:
                if points[customPointIndex].pointType == CORE_POINT_TYPE and points[
                    customPointIndex].group != point.group:
                    countGroupsNumber[point.group] += countGroupsNumber[points[customPointIndex].group]
                    del countGroupsNumber[points[customPointIndex].group]
                    mergeGroup(points, points[customPointIndex].group, point.group)
        # point.pointType >= 0 means it is BOUNDARY_POINT_TYPE
        elif point.pointType >= 0:
            corePointGroupIndex = points[point.pointType].group
            countGroupsNumber[corePointGroupIndex] += countGroupsNumber[point.group]
            del countGroupsNumber[point.group]
            point.group = corePointGroupIndex
    countGroupsNumber = sorted(countGroupsNumber.items(), key=lambda group: group[1], reverse=True)
    count = 0
    for key, _ in countGroupsNumber:
        count += 1
        for point in points:
            if point.group == key:
                point.group = -1 * count
        if count >= clusterCenterNumber:
            break


def showClusterAnalysisResults(points):
    colorStore = ['or', 'og', 'ob', 'oc', 'om', 'oy', 'ok']
    pylab.figure(figsize=(9, 9), dpi=80)
    for point in points:
        color = ''
        if point.group < 0:
            color = colorStore[-1 * point.group - 1]
        else:
            color = colorStore[-1]
        pylab.plot(point.x, point.y, color)
    pylab.show()


def main():
    clusterCenterNumber = 4
    pointsNumber = 500
    radius = 10
    Eps = 2
    minPointsNumber = 18
    points = generatePoints(pointsNumber, radius)
    pointsIndexGroupWithinBoundary = getPointsNumberWithinBoundary(points, Eps)
    decidePointsType(points, pointsIndexGroupWithinBoundary, minPointsNumber)
    dbscan(points, pointsIndexGroupWithinBoundary, clusterCenterNumber)
    showClusterAnalysisResults(points)


main()
