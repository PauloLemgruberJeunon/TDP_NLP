from tkinter import *


class VectorDrawGui:

    def __init__(self, master, cooc_matrix):
        self.master = master
        master.title('Inputs for drawing the word vectors')
        self.cooc_matrix = cooc_matrix

        self.status_var = StringVar()
        self.status_var.set('Status: Ok')
        self.curr_status_label = Label(master, textvariable=self.status_var, font=('TkDefaultFont', 11))

        self.noun1_label = Label(master, text='Enter the name of the first noun (first vector)',
                                 font=('TkDefaultFont', 11))
        self.noun2_label = Label(master, text='Enter the name of the second noun (second vector)',
                                 font=('TkDefaultFont', 11))
        self.verb1_label = Label(master, text='Enter the name of the first verb (x axis)', font=('TkDefaultFont', 11))
        self.verb2_label = Label(master, text='Enter the name of the second verb (y axis)', font=('TkDefaultFont', 11))

        self.noun1_entry = Entry(master, font=('TkDefaultFont', 11))
        self.noun2_entry = Entry(master, font=('TkDefaultFont', 11))
        self.verb1_entry = Entry(master, font=('TkDefaultFont', 11))
        self.verb2_entry = Entry(master, font=('TkDefaultFont', 11))

        self.submit_button = Button(master, text='Draw graph', command=self.submit_button, font=('TkDefaultFont', 11))

        self.use_filtered_matrix = IntVar()
        self.if_filtered_checkbox = Checkbutton(master, text='Use filtered co-occurrence matrix', relief=RAISED,
                                                variable=self.use_filtered_matrix, font=('TkDefaultFont', 11))
        self.if_filtered_checkbox.select()

        self.quit_button = Button(master, text='Exit Program', relief=RAISED, command=self.quit_program,
                                  font=('TkDefaultFont', 11))

        self.curr_status_label.grid(row=0, columnspan=2, sticky=W+E, padx=(0, 8), pady=8)

        self.noun1_label.grid(row=1, sticky=W+E, columnspan=2, padx=8)
        self.noun1_entry.grid(row=2, sticky=W+E, columnspan=2, padx=8, pady=(0, 15))

        self.noun2_label.grid(row=3, sticky=W+E, columnspan=2, padx=8)
        self.noun2_entry.grid(row=4, sticky=W+E, columnspan=2, padx=8, pady=(0, 15))

        self.verb1_label.grid(row=5, sticky=W+E, columnspan=2, padx=8)
        self.verb1_entry.grid(row=6, sticky=W+E, columnspan=2, padx=8, pady=(0, 15))

        self.verb2_label.grid(row=7, sticky=W+E, columnspan=2, padx=8)
        self.verb2_entry.grid(row=8, sticky=W+E, columnspan=2, padx=8, pady=(0, 15))

        self.submit_button.grid(row=9, column=0, padx=8, pady=8)
        self.if_filtered_checkbox.grid(row=9, column=1, padx=8, pady=8)

        self.quit_button.grid(row=10, sticky=W+E, columnspan=2, padx=8, pady=8)

    def submit_button(self):
        status = self.cooc_matrix.plot_two_word_vectors(self.noun1_entry.get(), self.noun2_entry.get(),
                                                        self.verb1_entry.get(), self.verb2_entry.get(),
                                                        self.use_filtered_matrix.get())

        if status:
            self.status_var.set('Status: Ok')
            self.curr_status_label.config(foreground='black')
        else:
            self.status_var.set('Status: One or more of the inputs where not found.\n Please check the xlsx archive.')
            self.curr_status_label.config(foreground='red')

    def quit_program(self):
        quit()
