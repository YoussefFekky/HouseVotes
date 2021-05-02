import pandas
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
from matplotlib import pyplot

# Read and split the dataset into attributes and classes
dataset = pandas.read_csv("house-votes-84.data.txt", header = None)
attributes = dataset.drop(0, axis = 1)
classes = dataset[0]

# Replace all '?'s in the data with the majority of votes for each attribute
for i in range(1, 17):
    nYes, nNo = 0, 0
    for j in range(len(attributes)):
        if attributes[i][j] == 'y':
            nYes += 1
            attributes[i][j] = 1
        elif attributes[i][j] == 'n':
            nNo += 1
            attributes[i][j] = 0
    moreYesses = nYes > nNo
    for j in range(len(attributes)):
        if attributes[i][j] == '?':
            if moreYesses:
                attributes[i][j] = 1
            else:
                attributes[i][j] = 0

# Run the experiment 6 times
print("~Training set size = 25%~\n")
for i in range(1, 7):
    print("Experiment", i)
    # Split the data into training and testing sets
    attributesTrain, attributesTest, classesTrain, classesTest = model_selection.train_test_split(attributes, classes, test_size=0.75)
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(attributesTrain, classesTrain)
    prediction = classifier.predict(attributesTest)

    # Measure the accuracy of the decision tree
    print("Tree size =", classifier.tree_.node_count)
    print("Accuracy =", metrics.accuracy_score(classesTest, prediction), '\n')

# Measure the mean, minimum and maximum accuracies bases on different training and testing splits
split = 0.7
accuracies, sizes = list(), list()
splits = list()
while split >= 0.3:
    splits.append("{}%".format(int((1-split)*100)))
    print("~Training set size = {}%~\n".format(int((1-split)*100)))
    meanSize, minSize, maxSize = 0, 1, 0
    meanAccuracy, minAccuracy , maxAccuracy = 0, 1, 0
    for i in range(5):
        # Split the data into training and testing sets
        attributesTrain, attributesTest, classesTrain, classesTest = model_selection.train_test_split(attributes, classes, test_size=split)
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(attributesTrain, classesTrain)
        prediction = classifier.predict(attributesTest)

        # Measure the accuracy of the decision tree
        treeSize = classifier.tree_.node_count
        accuracy = metrics.accuracy_score(classesTest, prediction)
        
        meanSize += treeSize
        meanAccuracy += accuracy

        if treeSize < minSize:
            minSize = treeSize
        if treeSize > maxSize:
            maxSize = treeSize

        if accuracy < minAccuracy:
            minAccuracy = accuracy
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
    meanSize /= 5
    meanAccuracy /= 5
    print("Mean tree size =", meanSize, "\nMinimum tree size =", minSize, "\nMaximum tree size =", maxSize, '\n')
    print("Mean accuracy =", meanAccuracy, "\nMinimum accuracy =", minAccuracy, "\nMaximum accuracy =", maxAccuracy, '\n')
    sizes.append(meanSize)
    accuracies.append(meanAccuracy)
    split -= 0.1

pyplot.plot(splits, accuracies)
pyplot.show()
pyplot.plot(splits, sizes)
pyplot.show()
