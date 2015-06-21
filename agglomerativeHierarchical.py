import math
import collections
import random
import copy
import pylab

try:
	import psyco
	psyco.full()
except ImportError:
	pass

FLOAT_MAX = 1e100

class Point:
	__slots__ = ["x", "y", "group"]
	def __init__(self, x = 0, y = 0, group = 0):
		self.x, self.y, self.group = x, y, group

def generatePoints(pointsNumber, radius):
	points = [Point() for _ in xrange(4 * pointsNumber)]
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

def getDistanceMap(points):
	distanceMap = {}
	for i in xrange(len(points)):
		for j in xrange(i + 1, len(points)):
			distanceMap[str(i) + '#' + str(j)] = solveDistanceBetweenPoints(points[i], points[j])
	distanceMap = sorted(distanceMap.iteritems(), key=lambda dist:dist[1], reverse=False)
	return distanceMap

def agglomerativeHierarchicalClustering(points, distanceMap, mergeRatio, clusterCenterNumber):
	unsortedGroup = {index: 1 for index in xrange(len(points))}
	for key, _ in distanceMap:
		lowIndex, highIndex = int(key.split('#')[0]), int(key.split('#')[1])
		if points[lowIndex].group != points[highIndex].group:
			lowGroupIndex = points[lowIndex].group
			highGroupIndex = points[highIndex].group
			unsortedGroup[lowGroupIndex] += unsortedGroup[highGroupIndex]
			del unsortedGroup[highGroupIndex]
			for point in points:
				if point.group == highGroupIndex:
					point.group = lowGroupIndex
		if len(unsortedGroup) <= int(len(points) * mergeRatio):
			break
	sortedGroup = sorted(unsortedGroup.iteritems(), key=lambda group: group[1], reverse=True)
	topClusterCenterCount = 0
	print sortedGroup, len(sortedGroup)
	for key, _ in sortedGroup:
		topClusterCenterCount += 1
		for point in points:
			if point.group == key:
				point.group = -1 * topClusterCenterCount
		if topClusterCenterCount >= clusterCenterNumber:
			break
	return points


def showClusterAnalysisResults(points):
	colorStore = ['or', 'og', 'ob', 'oc', 'om', 'oy', 'ok']
	pylab.figure(figsize=(9, 9), dpi = 80)
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
	mergeRatio = 0.025
	points = generatePoints(pointsNumber, radius)
	distanceMap = getDistanceMap(points)
	points = agglomerativeHierarchicalClustering(points, distanceMap, mergeRatio, clusterCenterNumber)
	showClusterAnalysisResults(points)

main()
