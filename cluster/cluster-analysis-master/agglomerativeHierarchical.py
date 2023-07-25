import math
import random
import pylab
import matplotlib.pyplot as plt
import seaborn as sns

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
    points = [Point() for _ in range(4 * pointsNumber)]
    originX = [-radius, -radius, radius, radius]
    originY = [-radius, radius, -radius, radius]
    count = 0
    countCenter = 0  # 象限
    for index, point in enumerate(points):
        count += 1
        r = random.random() * radius
        angle = random.random() * 2 * math.pi
        # 极坐标化，使随机点均匀分布
        point.x = r * math.cos(angle) + originX[countCenter]
        point.y = r * math.sin(angle) + originY[countCenter]
        point.group = index
        # point.group = random.randint(1, 4)  # 为每个点设置一个随机的分组（1~4）
        if count >= pointsNumber * (countCenter + 1):
            countCenter += 1
    return points


def visible(points):
    # 提取横坐标、纵坐标和组别信息
    x_values = [point.x for point in points]
    y_values = [point.y for point in points]
    groups = [point.group for point in points]

    # 设置 seaborn 风格
    sns.set(style="whitegrid")

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_values, y=y_values, hue=groups, palette='viridis', legend='full', s=70)
    plt.title('Generated Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(title='Group', loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def solveDistanceBetweenPoints(pointA, pointB):
    return (pointA.x - pointB.x) * (pointA.x - pointB.x) + (pointA.y - pointB.y) * (pointA.y - pointB.y)


def getDistanceMap(points):
    distanceMap = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distanceMap[str(i) + '#' + str(j)] = solveDistanceBetweenPoints(points[i], points[j])
    distanceMap = sorted(distanceMap.items(), key=lambda dist: dist[1], reverse=False)  # 字典的键值对转换为元组，按照距离从小到大排序
    return distanceMap


def agglomerativeHierarchicalClustering(points, distanceMap, mergeRatio, clusterCenterNumber):
    unsortedGroup = {index: 1 for index in range(len(points))}
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
    sortedGroup = sorted(unsortedGroup.items(), key=lambda group: group[1], reverse=True)
    topClusterCenterCount = 0
    print(sortedGroup, len(sortedGroup))
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
    mergeRatio = 0.025
    points = generatePoints(pointsNumber, radius)
    distanceMap = getDistanceMap(points)
    points = agglomerativeHierarchicalClustering(points, distanceMap, mergeRatio, clusterCenterNumber)
    showClusterAnalysisResults(points)


def test():
    pointsNumber = 100
    radius = 5
    points = generatePoints(pointsNumber, radius)
    visible(points)


def choose(choice):
    if choice == 0:
        main()
    else:
        test()


# choice = input("0:聚类 1:测试\n")
# choose(choice)

if __name__ == '__main__':
    main()
