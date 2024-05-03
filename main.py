from classifier.gui import Table
from classifier.naive_bayes import Naive_Bayes


def main():

    bayes = Naive_Bayes()
    bayes.run_naive_bayes()

    classes = ["Negative", "Positive"]
    for i in range(len(bayes.classes_name)):
        print(
            bayes.classes_name[i],
            bayes.predicted[i],
            bayes.classes_name[i] == bayes.predicted[i],
        )
    print("Accuracy: %0.2f" % bayes.accuracy)

    tb = Table(
        class_names=classes,
        col1=bayes.classes_name,
        col2=bayes.predicted,
        value=bayes.accuracy,
    )
    tb.create_Gui()


if __name__ == "__main__":
    main()
