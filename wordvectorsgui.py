from tkinter import *

class vector_draw_gui:
    def __init__(self, master, cooc_matrix):
        self.master = master
        master.title('Inputs for drawing the word vectors')
        self.cooc_matrix = cooc_matrix

        self.noun1_label = Label(master, text='Enter the name of the first noun (first vector)')
        self.noun2_label = Label(master, text='Enter the name of the second noun (second vector)')
        self.verb1_label = Label(master, text='Enter the name of the first verb (x axis)')
        self.verb2_label = Label(master, text='Enter the name of the second verb (y axis)')

        self.noun1_entry = Entry(master)
        self.noun2_entry = Entry(master)
        self.verb1_entry = Entry(master)
        self.verb2_entry = Entry(master)

        self.submit_button = Button(master, text='Draw graph', command=self.submit_button)

        self.use_filtered_matrix = IntVar()
        self.if_filtered_checkbox = Checkbutton(master, text='Use filtered co-occurrence matrix', relief=RAISED,
                                                variable=self.use_filtered_matrix)
        self.if_filtered_checkbox.select()

        self.quit_button = Button(master, text='Exit Program', relief=RAISED, command=self.quit_program)

        self.noun1_label.grid(row=0, sticky=W+E, columnspan=2, padx=8)
        self.noun1_entry.grid(row=1, sticky=W+E, columnspan=2, padx=8)

        self.noun2_label.grid(row=3, sticky=W+E, columnspan=2, padx=8)
        self.noun2_entry.grid(row=4, sticky=W+E, columnspan=2, padx=8)

        self.verb1_label.grid(row=6, sticky=W+E, columnspan=2, padx=8)
        self.verb1_entry.grid(row=7, sticky=W+E, columnspan=2, padx=8)

        self.verb2_label.grid(row=9, sticky=W+E, columnspan=2, padx=8)
        self.verb2_entry.grid(row=10, sticky=W+E, columnspan=2, padx=8)

        self.submit_button.grid(row=12, column=0, padx=8, pady=8)
        self.if_filtered_checkbox.grid(row=12, column=1, padx=8, pady=8)

        self.quit_button.grid(row=14, sticky=W+E, columnspan=2, padx=8, pady=8)

    def submit_button(self):
        self.cooc_matrix.plot_two_word_vectors(self.noun1_entry.get(), self.noun2_entry.get(), self.verb1_entry.get(),
                                          self.verb2_entry.get(), self.use_filtered_matrix.get())

    def quit_program(self):
        quit()


