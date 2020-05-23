import tkinter as tk
from tkinter import ttk
from main import *


class Table:
    def __init__(self, class_names=[], col1=[], col2=[], value=0):
        self.col1 = col1
        self.col2 = col2
        self.value = value
        self.class_names = class_names
        self.scores = tk.Tk()
        self.scores.resizable(False, False)
        self.cols = ('No.', 'Expected', 'Actual')
        self.listBox = ttk.Treeview(
            self.scores, columns=self.cols, show='headings', height=22)

    def display_Table(self):
        classes = {i: self.class_names[i]
                   for i in range(0, len(self.class_names))}
        # if not result:
        #     self.listBox.insert("", "end", values=(
        #         '-', '-', '-', '-', '-'))
        for i, ans in enumerate(self.col2):
            tg = 'false'
            if int(self.col1[i]) == int(ans):
                tg = 'true'
            self.listBox.insert("", "end", tags=(tg,), values=(
                i+1, classes[int(self.col1[i])], classes[int(ans)]))
        self.listBox.tag_configure('false', background='#FFE0DE')
        self.listBox.tag_configure('true', background='#E2FFDC')

    def create_Gui(self):
        self.scores.geometry("400x570")
        self.scores.title('Vector Space Model')

        self.label = tk.Label(self.scores, text="Parkinson's Disease", font=(
            "Arial", 30)).grid(row=0, columnspan=2)

        # answer = tk.StringVar()
        # searchQuery = tk.Entry(
        #     self.scores, width=94, textvariable=answer).place(x=10, y=52)

        self.label = tk.Label(self.scores, text=f"").grid(
            row=1, column=1, pady=5)

        self.label = tk.Label(self.scores, text=f"{self.value:0.2f}%",
                              font=("Arial", 14)).place(x=330, y=50)

        # self.label = tk.Label(self.scores, text="KNN Value:\t"+"3",font=("Arial", 10)).place(x=694+160, y=48-40)
        # self.label = tk.Label(self.scores, text="Training Data:\t" + str(int(indexer.param[0]*100))+"%",font=("Arial", 10)).place(x=694+160, y=48-20)
        # self.label = tk.Label(self.scores, text="Testing Data:\t" + str(int(100-indexer.param[0]*100))+"%",font=("Arial", 10)).place(x=694+160, y=48-0)

        # answer2 = tk.StringVar(value="0.0005")
        # searchQuery2 = tk.Entry(
        #     self.scores, width=10, textvariable=answer2, justify='right').place(x=710+205, y=50-35)

        vsb = ttk.Scrollbar(
            self.scores, orient="vertical", command=self.listBox.yview)
        vsb.place(x=385, y=79, height=460)
        vsb.configure(command=self.listBox.yview)
        self.listBox.configure(yscrollcommand=vsb.set)

        for col in self.cols:
            self.listBox.heading(col, text=col)
        self.listBox.grid(row=2, column=0, columnspan=2)
        self.listBox.column(self.cols[0], minwidth=40, width=60, stretch=tk.NO)
        self.listBox.column(self.cols[1], minwidth=75, width=160)
        self.listBox.column(self.cols[2], minwidth=75, width=160)
        # self.listBox.column(
        #     self.cols[1], minwidth=120, width=180, stretch=tk.NO)
        # self.listBox.column(self.cols[2], minwidth=120, width=522)
        # self.listBox.column(self.cols[5], minwidth=20, width=20)

        """ showScores = tk.Button(self.scores, text="Search", width=22,
                               command=lambda:
                               [
                                   self.listBox.delete(
                                       *self.listBox.get_children()),
                                   tb.display_Table(
                                       indexer.calculate(answer.get(), float(answer2.get())))
                               ]).place(x=798, y=45) """
        closeButton = tk.Button(self.scores, text="Close", width=15,
                                command=exit).grid(row=4, column=0, columnspan=2)

        self.display_Table()
        self.scores.mainloop()


if __name__ == "__main__":

    bayes = Naive_Bayes()
    bayes.run_naive_bayes()

    classes = ["Negative", "Positive"]
    for i in range(len(bayes.classes_name)):
        print(bayes.classes_name[i], bayes.predicted[i],
              bayes.classes_name[i] == bayes.predicted[i])
    print('Accuracy: %0.2f' % bayes.accuracy)

    tb = Table(class_names=classes, col1=bayes.classes_name,
               col2=bayes.predicted, value=bayes.accuracy)
    tb.create_Gui()
