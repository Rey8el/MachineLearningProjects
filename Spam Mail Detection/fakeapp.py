import pickle 

with open(r"D:\GIT\MachineLearningProjects\Spam Mail Detection\SpamDetectionModel.pkl", "rb") as file:
    classifier = pickle.load(file)

with open(r"D:\GIT\MachineLearningProjects\Spam Mail Detection\vectorizer.pkl", "rb") as file:
    cv = pickle.load(file)

def main():
    text = input("Enter the text here: ")
    vectorized = cv.transform([text])
    result = classifier.predict(vectorized)
    if result == 1:
        print("SPAM")
    else:
        print("NOT SPAM")

main()
