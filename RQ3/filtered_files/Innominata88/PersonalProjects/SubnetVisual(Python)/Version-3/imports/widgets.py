import tkinter as tk
import tkinter.font as tkfont
import tkinter.ttk as ttk
from dataclasses import dataclass

@dataclass
class WidgetStateColors:
    background: str
    text: str
    accent: str


@dataclass
class WidgetFonts:
    heading: tuple
    label: tuple
    entry: tuple
    tiny: tuple


@dataclass
class BorderWidths:
    left: str | float
    right: str | float
    top: str | float
    bottom: str | float


@dataclass
class WidgetStyle:
    enabled: WidgetStateColors
    disabled: WidgetStateColors
    readonly: WidgetStateColors
    borderwidths: BorderWidths
    fonts: WidgetFonts
    background: str


@dataclass
class ButtonStyle:
    label: str
    value: str | float | int
    unselected: WidgetStateColors
    selected: WidgetStateColors
    #disabled: WidgetStateColors


@dataclass
class SegmentedButtonStyle:
    buttons: list[ButtonStyle]
    fonts: WidgetFonts
    background: str
    borderwidths: BorderWidths



# Adapted from https://code.activestate.com/recipes/580798-tkinter-frame-with-different-border-sizes/?in=user-4189907
class BorderedFrame(tk.Frame):
    def __init__(self, 
                 parent, 
                 interior: tk.Widget = tk.Frame,
                 bordercolor: str = None, 
                 borderwidths: BorderWidths = BorderWidths(0, 0, 0, 0),
                 **kwargs):
        tk.Frame.__init__(self, parent, background=bordercolor)
        self.interior = interior(self, **kwargs)
        self.interior.grid(row=0, column=0, 
                           padx=(borderwidths.left, borderwidths.right), 
                           pady=(borderwidths.top, borderwidths.bottom),
                           sticky="NSEW")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)



class FixedSizeFramedLabel(tk.Frame):
    def __init__(self, 
                 parent, 
                 width: str | float = 24, 
                 height: str | float = 24, 
                 **kwargs):
        # Create outer frame and prevent resizing
        tk.Frame.__init__(self, parent, width=width, height=height)
        self.grid_propagate(0)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Create the label in the outer frame
        self.label = tk.Label(self, **kwargs)
        self.label.config(anchor="center", justify="center")
        
        # Adjust bottom padding to fix vertical text centering in cell
        self.label.grid(row=0, column=0, sticky="WENS", padx=0, 
                        pady=self._get_vertical_padding(), 
                        ipadx=0, ipady=0)

        # Color the background of the outer frame to account for label padding
        super().configure(bg=self.label.cget("bg"))


    # Overrides the default config to ensure the background colors match and label padding is correct
    def configure(self, cnf=None, **kwargs):
        if not cnf:
            cnf = {}
        cnf.update(**kwargs)

        bg = None

        if "bg" in cnf:
            bg = cnf.pop("bg")
        if "background" in cnf:
            bg = cnf.pop("background")

        if bg:
            super().configure(bg=bg)
            self.label.configure(bg=bg)

        if "height" in cnf:
            super().configure(height=cnf.pop("height"))
        if "width" in cnf:
            super().configure(width=cnf.pop("width"))
        if "font" in cnf:
            self.label.configure(font=cnf.pop("font"))
            self.label.grid(pady=self._get_vertical_padding())

        self.label.configure(cnf)


    def config(self, cnf=None, **kwargs):
        self.configure(cnf, **kwargs)


    def _get_vertical_padding(self):
        try:
            pady_bottom = (int(self.cget("height")) - int(self.label.cget("font").split()[-1])) / 2
        except ValueError as v:
            pady_bottom = (int(self.cget("height")) - int(tkfont.nametofont("TkDefaultFont").actual("size"))) / 2

        return (0, pady_bottom if pady_bottom > 0 else 0)



class CombFrame(tk.Frame):
     def __init__(self, 
                  parent = None,
                  widget: tk.Widget = tk.Entry,
                  varslist: list[tk.Variable] = None,
                  textlist: list[str] = None,
                  bordercolor: str = None,
                  outerborderwidths: BorderWidths = BorderWidths(1, 1, 1, 1), # Left, Right, Top, Bottom
                  innerborderwidth: int = 1,
                  width: str | float = 1,
                  **kwargs):
        tk.Frame.__init__(self, parent)
        self.cells = []
        
        # Outer container frame with the outside borders
        self.outer = BorderedFrame(self, bordercolor=bordercolor, borderwidths=outerborderwidths)
        self.outer.grid(row=0, column=0, sticky="WENS")

        # Inner container frame to hold the comb
        self.interior = tk.Frame(self.outer)
        self.interior.grid(row=0, column=0,
                           padx=(outerborderwidths.left, outerborderwidths.right), 
                           pady=(outerborderwidths.top, outerborderwidths.bottom),
                           sticky="NSEW")
        
        # Generate labels for the comb and place them
        if varslist:
            number_of_vars = len(varslist)
            for i in range(number_of_vars):
                cell = BorderedFrame(self.interior, interior=widget, bordercolor=bordercolor, 
                                    textvariable=varslist[i], 
                                    borderwidths=BorderWidths(innerborderwidth if i != 0 else 0, 0, 0, 0),
                                    width=width, **kwargs)
                self.cells.append(cell)
                cell.grid(column = i, row = 0, sticky="NSEW")
        else:
            number_of_vars = len(textlist)
            for i in range(number_of_vars):
                cell = BorderedFrame(self.interior, interior=widget, bordercolor=bordercolor, 
                                    text=textlist[i], 
                                    borderwidths=BorderWidths(innerborderwidth if i != 0 else 0, 0, 0, 0),
                                    width=width, **kwargs)
                self.cells.append(cell)
                cell.grid(column = i, row = 0, sticky="NSEW")


class LabeledEntryFrame(tk.Frame):
    def __init__(self, parent=None, label_text="Label", styles=WidgetStyle, textvariable=tk.Variable, **kwargs):
        super().__init__(parent, background=styles.background)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.style = styles
        self.var = textvariable
        self.label_text = tk.StringVar(value=label_text)

        self.tooltip = None

        self.inner = BorderedFrame(self, bordercolor=self.style.enabled.accent, borderwidths=self.style.borderwidths)
        self.inner.grid(row=1, column=0, sticky="EW", padx=0, pady=0, columnspan=2)

        self.inner.interior.config(background=self.style.enabled.background)
        self.inner.rowconfigure(0, weight=1)
        self.inner.columnconfigure(0, weight=1)
        
        self.entry = tk.Entry(self.inner.interior, 
                              background=self.style.enabled.background, 
                              foreground=self.style.enabled.text,
                              disabledbackground=self.style.disabled.background,
                              disabledforeground=self.style.disabled.text,
                              readonlybackground=self.style.readonly.background,
                              highlightbackground=self.style.enabled.background,
                              textvariable=self.var,
                              highlightthickness=0, bd=0, **kwargs)
        self.entry.grid(row=0, column=0, sticky="ew", padx=5, pady=5, columnspan=2)
        self.inner.interior.rowconfigure(0, weight=1)
        self.inner.interior.columnconfigure(0, weight=1)

        self.label = tk.Label(self, textvariable=self.label_text, font=self.style.fonts.label, foreground=self.style.enabled.accent, background=self.style.background)
        self.label.grid(row=0, column=0, columnspan=2, padx=0, sticky="SW")

        self.clear_button = tk.Label(self.inner.interior, background=self.style.enabled.background, text="X", font=self.style.fonts.tiny, foreground=self.style.enabled.accent)
        self.clear_button.grid(row=0, column=3, padx=2, pady=2)

        self.clear_button.bind("<Button-1>", self.clear_entry)

    def disable_entry(self):
        '''
        Sets the state of the entry widget to "disabled" and sets appropriate colors
        '''
        self.entry.config(state="disabled")
        self.label.config(foreground=self.style.disabled.accent)
        self.inner.config(background=self.style.disabled.accent)
        self.clear_button.config(foreground=self.style.disabled.accent, background=self.style.disabled.background)
        self.clear_button.unbind("<Button-1>")


    def readonly_entry(self):
        '''
        Sets the state of the entry widget to "readonly" and sets appropriate colors
        '''
        self.entry.config(state="readonly", foreground=self.style.readonly.text)
        self.label.config(foreground=self.style.readonly.accent)
        self.inner.config(background=self.style.readonly.accent)
        self.clear_button.config(foreground=self.style.readonly.background, background=self.style.readonly.background)
        self.clear_button.unbind("<Button-1>")


    def enable_entry(self):
        '''
        Sets the state of the entry widget to "normal" and sets appropriate colors
        '''
        self.entry.config(state="normal", foreground=self.style.enabled.text)
        self.label.config(foreground=self.style.enabled.accent)
        self.inner.config(background=self.style.enabled.accent)
        self.clear_button.config(foreground=self.style.enabled.accent, background=self.style.enabled.background)
        self.clear_button.bind("<Button-1>", self.clear_entry)


    def set_label_text(self, label_text):
        '''
        Sets the label text of the widget
        '''
        self.label_text.set(label_text)


    def clear_entry(self, event):
        '''
        Clears the entry text when the user clicks the "X" button
        '''
        self.entry.delete(0, "end")


class SegmentedButtonFrame(tk.Frame):
    def __init__(self, parent=None, selection_variable: tk.Variable = None, button_styles: SegmentedButtonStyle = None):
        super().__init__(parent, background=button_styles.background)

        self.styles = button_styles
        self.selected = selection_variable
        self.tooltip = None
        self.buttons = []

        self.inner = tk.Frame(self, background=self.styles.background)
        self.inner.grid(row=0, column=0, sticky="NSEW", columnspan=2)
        
        self.inner.rowconfigure(0, weight=1)
        self.inner.columnconfigure(0, weight=1)

        num_buttons = len(button_styles.buttons)
        name = self.winfo_name()

        for i in range(num_buttons):
            button = BorderedFrame(self.inner, bordercolor=self.styles.buttons[i].unselected.accent, borderwidths=self.styles.borderwidths, interior=tk.Label, background=self.styles.buttons[i].unselected.background, foreground=self.styles.buttons[i].unselected.text, font=self.styles.fonts.label, text=self.styles.buttons[i].label)
            self.columnconfigure(i, weight=1, uniform=name)
            self.inner.columnconfigure(i, weight=1, uniform=name)
            
            self.buttons.append(button)

            button.grid(row=0, column=i, sticky="NSEW")
            button.interior.bind("<Button-1>", lambda event, value=self.styles.buttons[i].value: self._change_selection(event, value))
            self.selected.trace("w", lambda *args, button_index=i: self._color_buttons(*args, button_index=button_index))

    def _change_selection(self, event, value):
        if value == self.selected.get():
            return
        
        self.selected.set(value)

    def _color_buttons(self, *args, button_index):
        button_info = self.styles.buttons[button_index]
        # Active Button
        if button_info.value == self.selected.get():
            self.buttons[button_index].config(background=button_info.selected.accent)
            self.buttons[button_index].interior.config(background=button_info.selected.background, foreground=button_info.selected.text)
        else:
            self.buttons[button_index].config(background=button_info.unselected.accent)
            self.buttons[button_index].interior.config(background=button_info.unselected.background, foreground=button_info.unselected.text)


@dataclass
class PrefixLenSliderColors:
    background: str
    slider_color: str
    on_colors: (str, str, str, str, str)
    off_colors: (str, str, str, str, str)


class PrefixLengthSlider(tk.Frame):
    def __init__(self, parent=None, slider_radius=13, line_width=5, scale_length=200, colors: PrefixLenSliderColors = None, **kwargs):
        super().__init__(parent, background=colors.background, **kwargs)
        self.slider_radius = slider_radius
        self.line_width = line_width
        self.scale_length = scale_length
        self.padding = slider_radius * 2
        self.slider_center = 55

        self.max_x = self.scale_length + self.padding

        self.canvas = tk.Canvas(self, height=75, width=self.max_x + self.padding, background=colors.background, bd=0, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="EW")

        self.min_value = 0
        self.max_value = 32
        self.range = self.max_value - self.min_value
        self.scale_factor = self.scale_length / self.range

        self.major_interval = 8
        self.minor_interval = 1
        self.major_tick_length = 10
        self.minor_tick_length = 5

        self.slider = None
        self.slider_pos = tk.IntVar(value=0)
        self.is_dragging = False

        self.colors = colors

        self.draw_scale()

        self.canvas.bind("<Button-1>", self.move_slider)
        self.canvas.bind("<B1-Motion>", self.drag_slider)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drag)
        self.slider_pos.trace("w", lambda *args: self._update_value(*args))


    def draw_scale(self):
        self.canvas.delete("all")

        interval = self.scale_length / self.range
        for i in range(self.min_value, self.max_value):
            x = (i - self.min_value) * self.scale_factor + self.padding

            self.canvas.create_line(x, self.slider_center, x + interval, self.slider_center, fill=self.colors.off_colors[i // self.major_interval], width=self.line_width)

        if not self.slider:
            self.slider_pos.set(self.min_value)
            x = self.padding
        else:
            x = (self.slider_pos.get() - self.min_value) * self.scale_factor + self.padding

        y = self.slider_center
        current_color = self.colors.on_colors[(self.slider_pos.get() - 1) // self.major_interval]
        y_min = y - (self.line_width / 2)
        y_max = y + (self.line_width / 2)

        for i in range(self.min_value, self.max_value):
            x_1 = (i - self.min_value) * self.scale_factor + self.padding

            if x_1 > x:
                break

            self.canvas.create_rectangle(x_1, y_min, x_1 + interval, y_max, fill=self.colors.on_colors[i // self.major_interval], outline="")


        self.slider = self.canvas.create_oval(x - self.slider_radius, y - self.slider_radius,
                                              x + self.slider_radius, y + self.slider_radius,
                                              fill=self.colors.slider_color, outline=current_color, width=3)
        self.canvas.create_text(x, y, text=self.slider_pos.get(), font=('TkFixedFont', 10), fill=current_color)
        
        for i in range(self.min_value, self.max_value + 1):
            x = (i - self.min_value) * self.scale_factor + self.padding
            y = 35
            current_color = self.colors.on_colors[(i - 1) // self.major_interval] if i == self.slider_pos.get() else self.colors.off_colors[(i - 1) // self.major_interval]
            
            if i % self.major_interval == 0:
                self.canvas.create_line(x, y - self.major_tick_length, x, y, fill=current_color)
                y -= (20 + self.major_tick_length)
                self.canvas.create_text(x, y, text=str(i), font=('TkFixedFont', 10), anchor="n", fill=current_color)
            elif i % self.minor_interval == 0:
                self.canvas.create_line(x, y - self.minor_tick_length, x, y, fill=current_color)
        

    def move_slider(self, event):
        x, y = event.x, event.y
        slider_coordinates = self.canvas.coords(self.slider)
        if slider_coordinates[0] <= x <= slider_coordinates[2] and slider_coordinates[1] <= y <= slider_coordinates[3]:
            self.is_dragging = True
        else:
            x = max(self.padding, min(x, self.max_x))
            self.slider_pos.set(int(((x - self.padding) / self.scale_factor) + self.min_value))
            self.draw_scale() 


    def drag_slider(self, event):
        if self.is_dragging:
            x = event.x
            x = max(self.padding, min(x, self.max_x))
            self.slider_pos.set(int(((x - self.padding) / self.scale_factor) + self.min_value))
            self.draw_scale()


    def stop_drag(self, event):
        self.is_dragging = False

    
    def _update_value(self, *args):
        self.draw_scale()





class Tooltip:
    '''
    Class generated by ChatGPT and slightly modified by me
    Creates tooltips to link to widgets
    '''
    def __init__(self, widget, text, background="systemTextBackgroundColor", foreground="systemTextColor"):
        self.widget = widget
        self.text = text
        self.background = background
        self.foreground = foreground
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip, text=self.text, background=self.background, foreground=self.foreground)
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

    def change_text(self, text):
        self.text = text