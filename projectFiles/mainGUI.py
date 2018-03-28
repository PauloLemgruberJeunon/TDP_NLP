from tkinter import *
from tkinter import messagebox

import os
import threading

import projectFiles.main as main_func
import projectFiles.constants as cts


file_not_found_status = 'The file was not found'
ok_status = 'Ok'


class MainGUI(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.path_to_txt = cts.path_to_txtFolder
        self.path_to_xlsx = cts.path_to_generated_xlsx

        self.cooc_matrix_vector_plotter = None

        Grid.rowconfigure(self, 0, weight=1)
        Grid.columnconfigure(self, 0, weight=1)

        self.status_frame = Frame(self)
        self.status_frame.grid(row=0, sticky='nsew')
        self.status_frame.grid_rowconfigure(0, weight=1)
        self.status_frame.grid_columnconfigure((0, 1), weight=1)

        label = Label(self.status_frame, text='Status: ')
        label.grid(row=0, column=0, columnspan=1, sticky=E)

        self.status_label = Label(self.status_frame, text='Ok')
        self.status_label.grid(row=0, column=1, columnspan=1, sticky=W)

        self.container = Frame(self)
        self.container.grid(row=1, sticky='nsew')
        self.container.grid_rowconfigure(0, weight=1)
        self. container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for myFrame in (LoadTxtOrXlsxFrame, LoadTxt, LoadXlsx, LoadingScreen, VectorDrawGui):
            frame_name = myFrame.__name__
            frame = myFrame(parent=self.container, controller=self)
            self.frames[frame_name] = frame

            Grid.rowconfigure(frame, 0, weight=1)
            Grid.columnconfigure(frame, 0, weight=1)

        frame = self.frames['LoadTxtOrXlsxFrame']
        frame.grid(row=0, column=0, sticky='nsew')

    def show_frame(self, curr_frame_name, new_frame_name):
        self.frames[curr_frame_name].grid_forget()

        frame = self.frames[new_frame_name]
        frame.grid(row=0, column=0, sticky='nsew')

    def change_status(self, status):
        self.status_label['text'] = status

    def create_cooc_matrix_from_txt(self, curr_frame_name, txt_input_name, encoding_type_value, save_in_xlsx,
                                    workbook_name, enable_verb_filter, enable_lemmatization):

        self.show_frame(curr_frame_name, 'LoadingScreen')

        my_thread = CoocMatrixThread(main_func.load_from_txt, self.path_to_txt + txt_input_name, encoding_type_value,
                                     save_in_xlsx, self.path_to_xlsx + workbook_name, enable_verb_filter,
                                     enable_lemmatization)

        my_thread.start()
        my_thread.join()

        self.frames['VectorDrawGui'].cooc_matrix_vector_plotter = my_thread.get_cooc_matrix()

        self.show_frame('LoadingScreen', 'VectorDrawGui')

    def create_cooc_matrix_from_xlsx(self, curr_frame_name, workbook_name):
        self.show_frame(curr_frame_name, 'LoadingScreen')

        my_thread = CoocMatrixThread(main_func.load_from_xlsx, self.path_to_xlsx + workbook_name)

        my_thread.start()
        my_thread.join()

        self.frames['VectorDrawGui'].cooc_matrix_vector_plotter = my_thread.get_cooc_matrix()
        self.show_frame('LoadingScreen', 'VectorDrawGui')

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            quit(0)
            self.destroy()


class LoadTxtOrXlsxFrame(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        self.controller = controller

        label = Label(self, text="Choose the program input form")
        label.grid(row=0, sticky=W+E+S, pady=1)

        self.xlsx_button = Button(self, text='Input from xlsx',
                                  command=lambda: self.controller.show_frame(self.name(), "LoadXlsx"))
        self.txt_button = Button(self, text='Input from txt',
                                 command=lambda: self.controller.show_frame(self.name(), "LoadTxt"))

        self.txt_button.grid(row=1, sticky=W+E, pady=10, padx=15)
        self.xlsx_button.grid(row=2, sticky=W+E, pady=(0, 10), padx=15)

    @staticmethod
    def name():
        return 'LoadTxtOrXlsxFrame'


class LoadTxt(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        self.controller = controller

        self.label_1 = Label(self, text='Enter the name of the txt input file')
        self.label_2 = Label(self, text='The file must be on the \'txtFiles\' folder')
        self.label_1.grid(row=0, sticky=W+E+S, pady=(15, 0), columnspan=2)
        self.label_2.grid(row=1, sticky=W+E+S, pady=(0, 10), columnspan=2)

        self.txt_input_name_value = StringVar()
        self.txt_input_name_entry = Entry(self, textvariable=self.txt_input_name_value)
        self.txt_input_name_entry.grid(row=2, sticky=W+E, columnspan=2, padx=30)

        self.inside_frame = Frame(self)
        self.inside_frame.grid(row=3, columnspan=2, padx=30, pady=15)

        self.label_3 = Label(self.inside_frame, text='Select the text encoding: ')
        self.label_3.grid(row=0, column=0, sticky=E)

        self.encoding_type_value = StringVar()
        self.encoding_type_value.set('utf8')
        self.encoding_type_options = OptionMenu(self.inside_frame,
                                                self.encoding_type_value, "ascii", "unicode", "utf8", "utf16")
        self.encoding_type_options.grid(row=0, column=1, sticky=W)

        self.save_in_xlsx_value = BooleanVar()
        self.save_in_xlsx_value.set(False)
        self.save_in_xlsx_checkbutton = Checkbutton(self, text="Save the results in xlsx file?",
                                                    variable=self.save_in_xlsx_value, onvalue=True, offvalue=False)

        self.save_in_xlsx_checkbutton.grid(row=4, pady=15, sticky=W+E, columnspan=2)

        self.matrix_options_frame = Frame(self)
        self.matrix_options_frame.grid(row=5, columnspan=2, padx=30, pady=15)

        self.enable_verb_filter_value = BooleanVar()
        self.enable_verb_filter_value.set(True)
        self.enable_verb_filter_checkbutton = Checkbutton(self.matrix_options_frame, text='Enable verb filter',
                                                          variable=self.enable_verb_filter_value, onvalue=True,
                                                          offvalue=False)

        self.enable_verb_filter_checkbutton.grid(row=0, column=0, sticky=E, columnspan=1)

        self.enable_lemmatization_value = BooleanVar()
        self.enable_lemmatization_value.set(True)
        self.enable_lemmatization_checkbutton = Checkbutton(self.matrix_options_frame, text='Enable lemmatization',
                                                            variable=self.enable_lemmatization_value, onvalue=True,
                                                            offvalue=False)

        self.enable_lemmatization_checkbutton.grid(row=0, column=1, sticky=W, columnspan=1)

        self.label_4 = Label(self, text='Enter the name of the xlsx file', state=DISABLED)
        self.label_4.grid(row=6, pady=15, sticky=W+E, columnspan=2)

        self.workbook_name_value = StringVar()
        self.workbook_name_entry = Entry(self, textvariable=self.workbook_name_value, state=DISABLED)

        self.workbook_name_entry.grid(row=7, pady=(0, 15), columnspan=2, sticky=W+E, padx=30)

        self.submit_button = Button(self, text='Submit', command=self.submit_function)
        self.submit_button.grid(row=8, pady=(0, 15), padx=15, columnspan=2, sticky=W+E)

        self.back_button = Button(self, text='Go Back',
                                  command=lambda: self.controller.show_frame(self.name(), 'LoadTxtOrXlsxFrame'))
        self.back_button.grid(row=9, pady=(0, 10), padx=15, sticky=W+E)

        self.save_in_xlsx_checkbutton.config(command=self.checkbutton_pressed)

    @staticmethod
    def name():
        return 'LoadTxt'

    def checkbutton_pressed(self):
        if self.save_in_xlsx_value.get():
            self.label_4['state'] = 'normal'
            self.workbook_name_entry['state'] = 'normal'
        else:
            self.label_4['state'] = 'disabled'
            self.workbook_name_entry['state'] = 'disabled'

    def submit_function(self):
        success = check_for_file(self.controller.path_to_txt, self.txt_input_name_value.get(),
                                 self.controller.change_status)

        if not success:
            return

        processing_thread = NormalThread(self.controller.create_cooc_matrix_from_txt, self.name(),
                                         self.txt_input_name_value.get(), self.encoding_type_value.get(),
                                         self.save_in_xlsx_value.get(), self.workbook_name_value.get(),
                                         self.enable_verb_filter_value.get(), self.enable_lemmatization_value.get())

        processing_thread.start()


class LoadXlsx(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        self.controller = controller

        self.label_1 = Label(self, text='Enter the name of the xlsx file to load the content')
        self.label_1.grid(row=0, padx=15, pady=(15, 0), sticky=E+W+S)

        self.label_2 = Label(self, text='The file must be on the folder \'xlsxFiles\'')
        self.label_1.grid(row=1, padx=15, pady=(0, 10), sticky=E + W + S)

        self.xlsx_file_name_value = StringVar()
        self.xlsx_file_name_entry = Entry(self, textvariable=self.xlsx_file_name_value)
        self.xlsx_file_name_entry.grid(row=2, padx=15, pady=(0, 15), sticky=E+W)

        self.submit_button = Button(self, text='Submit', command=self.submit_function)
        self.submit_button.grid(row=3, pady=(0, 15), padx=15, sticky=W+E)

        self.back_button = Button(self, text='Go Back',
                                  command=lambda: self.controller.show_frame(self.name(), 'LoadTxtOrXlsxFrame'))
        self.back_button.grid(row=4, pady=(0, 10), padx=15, sticky=W+E)

    @staticmethod
    def name():
        return 'LoadXlsx'

    def submit_function(self):
        success = check_for_file(self.controller.path_to_xlsx, self.xlsx_file_name_value.get(),
                                 self.controller.change_status)

        if not success:
            return

        processing_thread = NormalThread(self.controller.create_cooc_matrix_from_xlsx, self.name(),
                                         self.xlsx_file_name_value.get())

        processing_thread.start()

class LoadingScreen(Frame):

    def __init__(self, parent, controller):
        Frame.__init__(self, parent)

        self.controller = controller

        label = Label(self, text='Loading', font=('TkDefaultFont', 30))
        label.grid(sticky=W+E)

    @staticmethod
    def name():
        return 'LoadingScreen'


class VectorDrawGui(Frame):

    def __init__(self, parent, controller, cooc_matrix_vector_plotter=None):
        Frame.__init__(self, parent)

        self.controller = controller
        controller.title('Inputs for drawing the word vectors')
        self.cooc_matrix_vector_plotter = cooc_matrix_vector_plotter

        self.status_var = StringVar()
        self.status_var.set('Status: Ok')
        self.curr_status_label = Label(self, textvariable=self.status_var, font=('TkDefaultFont', 11))

        self.noun1_label = Label(self, text='Enter the name of the first noun (first vector)',
                                 font=('TkDefaultFont', 11))
        self.noun2_label = Label(self, text='Enter the name of the second noun (second vector)',
                                 font=('TkDefaultFont', 11))
        self.verb1_label = Label(self, text='Enter the name of the first verb (x axis)', font=('TkDefaultFont', 11))
        self.verb2_label = Label(self, text='Enter the name of the second verb (y axis)', font=('TkDefaultFont', 11))

        self.noun1_entry = Entry(self, font=('TkDefaultFont', 11))
        self.noun2_entry = Entry(self, font=('TkDefaultFont', 11))
        self.verb1_entry = Entry(self, font=('TkDefaultFont', 11))
        self.verb2_entry = Entry(self, font=('TkDefaultFont', 11))

        self.submit_button = Button(self, text='Draw graph', command=self.submit_button, font=('TkDefaultFont', 11))

        self.quit_button = Button(self, text='Exit Program', relief=RAISED, command=self.quit_program,
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

        self.quit_button.grid(row=10, sticky=W+E, columnspan=2, padx=8, pady=8)
#product_design_and_development.txt
    @staticmethod
    def name():
        return 'VectorDrawGui'

    def submit_button(self):
        if self.cooc_matrix_vector_plotter is None:
            self.controller.change_status('cooc-matrix not generated yet')
            return

        status = self.cooc_matrix_vector_plotter(self.noun1_entry.get(), self.noun2_entry.get(),self.verb1_entry.get(),
                                                 self.verb2_entry.get())

        if status:
            self.status_var.set('Status: Ok')
            self.curr_status_label.config(foreground='black')
        else:
            self.status_var.set('Status: One or more of the inputs where not found.\n Please check the xlsx archive.')
            self.curr_status_label.config(foreground='red')

    @staticmethod
    def quit_program():
        quit()


class NormalThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self.target = target
        self.args = args

    def run(self):
        self.target(*self.args)


class CoocMatrixThread(threading.Thread):
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self.target = target
        self.args = args

        self.cooc_matrix_vector_plotter = None

    def run(self):
        self.cooc_matrix_vector_plotter = self.target(*self.args)

    def get_cooc_matrix(self):
        if self.cooc_matrix_vector_plotter is not None:
            return self.cooc_matrix_vector_plotter


def main():
    app = MainGUI()
    app.mainloop()


def check_for_file(path, file_name, update_status_func):
    print(path+file_name)
    if os.path.isfile(path + file_name):
        update_status_func(ok_status)
        return True
    else:
        update_status_func(file_not_found_status + ': ' + file_name)
        return False


if __name__ == "__main__":
    main()
