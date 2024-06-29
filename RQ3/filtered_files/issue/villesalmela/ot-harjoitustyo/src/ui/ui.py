from threading import Thread
from functools import wraps
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, PhotoImage, messagebox, StringVar

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates

from ui.figure_config import FigureConfig
from utils.utils import check_file


def with_loading_screen(func):
    "Generated with ChatGPT."

    @wraps(func)
    def wrapper(*args, **kwargs):

        # Function to run in a separate thread
        def run():
            result[0] = func(*args, **kwargs)
            overlay.destroy()

        # Function to stop the thread
        def stop_thread():
            if thread.is_alive():
                # Attempt to join the thread, timing out almost immediately
                thread.join(timeout=0.1)
            overlay.destroy()

        root = args[0]  # Assuming the first argument is always the Tkinter root

        # Create an overlay frame
        overlay = ttk.Frame(root)
        overlay.place(x=1, y=1, relwidth=1, relheight=1)

        # Inner frame to hold widgets
        inner_frame = ttk.Frame(overlay)
        inner_frame.pack(expand=True)

        # Add text
        loading_label = ttk.Label(inner_frame, text="Please wait...")
        loading_label.pack()

        # Button to cancel the operation
        cancel_button = ttk.Button(inner_frame, text="Cancel", command=stop_thread)
        cancel_button.pack()

        # This will store the function's result
        result = [None]

        # Start the thread
        thread = Thread(target=run, daemon=True)
        thread.start()
        overlay.wait_window()

        return result[0]

    return wrapper


class PcapUi(tk.Tk):

    def __init__(self, analyze_function):
        super().__init__()
        self.analyze_function = analyze_function

        self.title('PCAP Analyzer')
        self.geometry('1200x800')
        icon = PhotoImage(file='assets/icon.png')
        self.iconphoto(True, icon)

        self.text_areas = {}
        self.figures = {}
        self.canvases = {}
        self.plots = {}
        self.indicators = {}

        self.figure_id = 0
        self.plot_id = 0
        self.text_area_id = 0
        self.indicator_id = 0

        self.create_menu()
        self.initialize_ui()

    def create_menu(self):
        self.menu_bar = tk.Menu(self)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)

        menu_data = [
            ("Open File...", "Ctrl+O", "<Control-o>", self.open_file),
            ("Reset", "Ctrl+R", "<Control-r>", self.reset),
            ("Exit", "Ctrl+Q", "<Control-q>", self.close)
        ]

        for item, shortcut_frontend, shortcut_backend, command in menu_data:
            self.file_menu.add_command(
                label=item, accelerator=shortcut_frontend, command=command)
            self.bind(shortcut_backend, lambda event, cmd=command: cmd())

        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.config(menu=self.menu_bar)

    def initialize_ui(self):
        self.frame = ttk.Frame(self)
        self.frame.pack(padx=10, pady=10, fill='x', expand=False)

        self.reset_button = ttk.Button(self.frame, text="Reset", command=self.reset)
        self.reset_button.pack(side='right')

        # Create the notebook for tabs
        self.notebook = ttk.Notebook(self)

        # Create frames for tabs
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)

        # Add tabs to the notebook
        self.notebook.add(self.tab1, text='Overview')
        self.notebook.add(self.tab2, text='Details')
        self.notebook.add(self.tab3, text='DNS')

        # Pack to make visible
        self.notebook.pack(expand=True, fill=tk.BOTH)

        # Prepare tab 1
        indicators_frame = ttk.Frame(self.tab1)
        self.packet_count_id = self.create_indicator(indicators_frame, 0, "Packet Count", 0, 0)
        self.data_amount_id = self.create_indicator(indicators_frame, 0, "Data Amount", 0, 1)
        self.duration_id = self.create_indicator(indicators_frame, 0, "Duration", 0, 2)
        self.starttime_id = self.create_indicator(indicators_frame, 0, "Start Time", 0, 3)
        self.endtime_id = self.create_indicator(indicators_frame, 0, "End Time", 0, 4)
        indicators_frame.pack(expand=True)

        self.overview_figure_id = self.create_figure_and_canvas(self.tab1)
        self.speed_plot_id = self.create_plot(self.overview_figure_id, 111)

        # Prepare tab 2
        self.summary_text_area_id = self.create_scrollable_text_area(self.tab2)

        # Prepare tab 3
        figure_id = self.create_figure_and_canvas(self.tab3)

        self.dns_plot_id_1 = self.create_plot(figure_id, 211)
        self.dns_plot_id_2 = self.create_plot(figure_id, 212)

    def create_scrollable_text_area(self, container):

        # Assigning a unique ID to the text area
        text_area_id = self.text_area_id
        self.text_area_id += 1

        # Defining
        text_area = scrolledtext.ScrolledText(container, wrap=tk.WORD)

        # Store the text area
        self.text_areas[text_area_id] = text_area

        # Displaying
        self.text_areas[text_area_id].pack(
            padx=10, pady=10, fill=tk.BOTH, expand=True)

        return text_area_id

    def create_indicator(self, parent, value, title, row, column):
        "Generated with ChatGPT."

        # Assigning a unique ID to the indicator
        indicator_id = self.indicator_id
        self.indicator_id += 1

        # Create StringVar to hold the value
        value_holder = StringVar(parent, value)

        # Create a new Frame as a child of the specified parent widget
        frame = ttk.Frame(parent, borderwidth=2, relief="groove")

        # Create a Label for the number, large and bold
        number_label = ttk.Label(frame, textvariable=value_holder, font=('Helvetica', 24, 'bold'))
        number_label.pack(pady=(0, 5))  # Add some padding below the number

        # Create a Label for the title, smaller and less prominent
        title_label = ttk.Label(frame, text=title, font=('Helvetica', 14))
        title_label.pack(pady=(5, 0))  # Add some padding above the title

        # Place the frame using the grid layout manager
        frame.grid(row=row, column=column, padx=10, pady=10, sticky='ew')

        # Store the frame
        self.indicators[indicator_id] = value_holder

        return indicator_id

    def create_figure_and_canvas(self, container):

        # Create a figure
        figure = plt.Figure()

        # Assigning a unique ID to the figure
        figure_id = self.figure_id
        self.figure_id += 1

        # Store the figure
        self.figures[figure_id] = figure

        # Create canvas for the figure
        canvas = FigureCanvasTkAgg(figure, container)

        # Store the canvas
        self.canvases[figure_id] = canvas

        # Display the canvas
        canvas_widget = self.canvases[figure_id].get_tk_widget()
        canvas_widget.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        # Return the ID
        return figure_id

    def create_plot(self, figure_id, position: int):

        # Assigning a unique ID to the plot
        plot_id = self.plot_id
        self.plot_id += 1

        # Fetch the parent figure
        figure = self.figures[figure_id]

        # Create the figure
        plot = figure.add_subplot(position)

        # Store the plot and its parent figure
        self.plots[plot_id] = plot, figure_id

        # Return the ID
        return plot_id

    def open_file(self):

        # Ask the user to select a file
        file_path = filedialog.askopenfilename()

        if file_path:  # file selected
            try:
                check_file(file_path)
            except FileNotFoundError as e:
                messagebox.showerror("Error", str(e))
                return

            result = self.process_file(file_path)

            if result is None:  # User cancelled the operation
                return
            produced_text, dns_domains, dns_servers, speed_config, indicators = result
            self.display_text(text_area_id=self.summary_text_area_id, text=produced_text)
            self.display_bar_graph(self.dns_plot_id_1, dns_domains)
            self.display_bar_graph(self.dns_plot_id_2, dns_servers)
            self.display_timeseries_dual(self.speed_plot_id, speed_config)
            self.display_indicator(self.packet_count_id, indicators["packet_count"])
            self.display_indicator(self.data_amount_id, indicators["data_amount"])
            self.display_indicator(self.duration_id, indicators["duration"])
            self.display_indicator(self.starttime_id, indicators["start_time"])
            self.display_indicator(self.endtime_id, indicators["end_time"])
        else:  # no file selected
            pass

    @with_loading_screen
    def process_file(self, file_path):
        return self.analyze_function(file_path)

    def display_indicator(self, indicator_id, value):
        self.indicators[indicator_id].set(value)

    def display_text(self, text_area_id, text):

        # Fetch the text area
        text_area = self.text_areas[text_area_id]

        # Make it read-write
        text_area.config(state=tk.NORMAL)

        # Resetting
        text_area.delete('1.0', tk.END)

        # Displaying
        text_area.insert(tk.END, text)

        # Make it read-only
        text_area.config(state=tk.DISABLED)

    def display_timeseries_dual(self, plot_id, speed_config: FigureConfig):

        # Fetch the plot and its parent figure
        plot, figure_id = self.plots[plot_id]

        # Unpacking the configuration
        title = speed_config.title
        data1 = speed_config.data1
        xlabel = speed_config.xlabel
        y1label = speed_config.y1label
        color1 = speed_config.color1
        data2 = speed_config.data2
        y2label = speed_config.y2label
        color2 = speed_config.color2

        # Setup general
        plot_color = color1
        plot.set_title(title)

        # Bail out if there's not enough data
        if len(data1) < 2:

            # Generated with ChatGPT
            plot.text(0.5, 0.5, "Not enough data to plot", horizontalalignment='center',
                      verticalalignment='center', transform=plot.transAxes, fontsize=14)

            # Draw the text
            self.canvases[figure_id].draw()
            return

        # Setup the x-axis
        plot.set_xlabel(xlabel)
        plot_locator = mdates.AutoDateLocator()
        plot_formatter = mdates.ConciseDateFormatter(plot_locator)

        plot.xaxis.set_major_locator(plot_locator)
        plot.xaxis.set_major_formatter(plot_formatter)

        duration = data1.index[-1] - data1.index[0]

        if duration <= pd.Timedelta(minutes=1):
            print("Setting minor ticks for seconds")
            minor_locator = mdates.SecondLocator()
        elif duration <= pd.Timedelta(hours=1):
            print("Setting minor ticks for minutes")
            minor_locator = mdates.MinuteLocator()
        elif duration <= pd.Timedelta(days=1):
            print("Setting minor ticks for hours")
            minor_locator = mdates.HourLocator()
        elif duration <= pd.Timedelta(days=30):
            print("Setting minor ticks for days")
            minor_locator = mdates.DayLocator()
        elif duration <= pd.Timedelta(days=365):
            print("Setting minor ticks for months")
            minor_locator = mdates.MonthLocator()
        else:
            print("Setting minor ticks for quarters")
            minor_locator = mdates.MonthLocator(interval=3)

        plot.xaxis.set_minor_locator(minor_locator)

        # Setup grid
        plot.grid(which='minor', linestyle='dashed', linewidth='0.5', color='gray')
        plot.grid(which='major', linestyle='solid', linewidth='1.0', color='black')

        # Setup y-axis
        plot.set_ylabel(y1label, color=plot_color)
        plot.fill_between(data1.index, data1, color=plot_color, alpha=0.4)
        plot.yaxis.grid(False)

        # Setup second plot
        plot2 = plot.twinx()
        plot2_color = color2
        plot2.set_ylabel(y2label, color=plot2_color)
        plot2.plot(
            data2.index,
            data2,
            color=plot2_color,
            linestyle='none',
            marker='o',
            markersize=1.5)

        # Syncronize the two axes
        freq = data1.index.freq
        freqstr = data1.index.freqstr

        if freq == pd.Timedelta('1s'):  # Force same axis for both
            ymin, ymax = plot2.get_ylim()
            plot2.set_ylim(0, ymax)
            plot.set_ylim(0, ymax)
        else:  # Dynamic, separate, positive axis
            plot2.set_ylim(bottom=0)
            plot.set_ylim(bottom=0)

        # Info to display in the infobox
        interval_length = freqstr
        interval_count = len(data1)

        # Adding an infobox (Generated with ChatGPT)
        info_text = f"Interval length: {interval_length}\nInterval count: {interval_count}"
        plot.text(
            0.05,
            0.95,
            info_text,
            transform=plot.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor='white',
                edgecolor='black',
                alpha=0.5))

        # Adjust layout to make room for the labels if necessary
        self.figures[figure_id].tight_layout()

        # Refreshing the canvas
        self.canvases[figure_id].draw()

    def display_bar_graph(
        self,
        plot_id: int,
        config: FigureConfig
    ):

        # Fetch the plot and its parent figure
        plot, figure_id = self.plots[plot_id]

        # Unpacking the configuration
        title = config.title
        data = config.data1
        xlabel = config.xlabel
        ylabel = config.y1label
        color = config.color1

        # Creating the horizontal bar graph
        keys = list(data.keys())
        values = list(data.values())
        bars = plot.barh(range(len(keys)), values, color=color, alpha=0.7)

        # Adding keys as labels on each bar
        for bar, key in zip(bars, keys):
            plot.text(0, bar.get_y() + bar.get_height() / 2, key,
                      va=tk.CENTER, ha=tk.LEFT, color='black')

        # Only keep integer ticks on the x-axis
        plot.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Hide the y-axis labels
        plot.set_yticks([])

        # Adding title and labels
        if title:
            plot.set_title(title)
        if xlabel:
            plot.set_xlabel(xlabel)
        if ylabel:
            plot.set_ylabel(ylabel)

        # Adjust layout to make room for the labels if necessary
        self.figures[figure_id].tight_layout()

        # Refreshing the canvas
        self.canvases[figure_id].draw()

    def close(self):
        self.destroy()

    def reset(self):
        # Resetting the text areas
        for text_area in self.text_areas.values():
            text_area.config(state=tk.NORMAL)
            text_area.delete('1.0', tk.END)
            text_area.config(state=tk.DISABLED)

        # Resetting the plots
        for plot, _ in self.plots.values():
            plot.clear()

        # Resetting the indicators
        for indicator in self.indicators.values():
            indicator.set(0)

        # Refreshing the canvas
        for canvas in self.canvases.values():
            canvas.draw()


def start_app(analyze_function):
    app = PcapUi(analyze_function)
    app.eval('tk::PlaceWindow . center')
    app.mainloop()
