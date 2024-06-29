import subprocess, os, sys, curses
"""
The following  get_*_*_str() functions were generated by chatGPT  just in case you're wondering.
"""

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    # Background colors
    BLACK_BG = '\033[40m'
    RED_BG = '\033[101m'
    GREEN_BG = '\033[102m'
    YELLOW_BG = '\033[103m'
    BLUE_BG = '\033[104m'
    MAGENTA_BG = '\033[105m'
    CYAN_BG = '\033[106m'
    WHITE_BG = '\033[107m'

def colored_text(text, color_code):
    return color_code + text + Colors.RESET

# Functions for colored texts
def get_red_str(text) -> str:
    return colored_text(text, Colors.RED)

def get_green_str(text) -> str:
    return colored_text(text, Colors.GREEN)

def get_yellow_str(text) -> str:
    return colored_text(text, Colors.YELLOW)

def get_blue_str(text) -> str:
    return colored_text(text, Colors.BLUE)

def get_magenta_str(text) -> str:
    return colored_text(text, Colors.MAGENTA)

def get_cyan_str(text) -> str:
    return colored_text(text, Colors.CYAN)

def get_white_str(text) -> str:
    return colored_text(text, Colors.WHITE)

# Functions for bold text in different colors
def get_bold_red_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.RED)

def get_bold_green_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.GREEN)

def get_bold_yellow_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.YELLOW)

def get_bold_blue_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.BLUE)

def get_bold_magenta_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.MAGENTA)

def get_bold_cyan_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.CYAN)

def get_bold_white_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.WHITE)

# Functions for underlined text in different colors
def get_underlined_red_str(text) -> str:
    return colored_text(text, Colors.UNDERLINE + Colors.RED)

def get_underlined_green_str(text) -> str:
    return colored_text(text, Colors.UNDERLINE + Colors.GREEN)

def get_underlined_yellow_str(text) -> str:
    return colored_text(text, Colors.UNDERLINE + Colors.YELLOW)

def get_underlined_blue_str(text) -> str:
    return colored_text(text, Colors.UNDERLINE + Colors.BLUE)

def get_underlined_magenta_str(text) -> str:
    return colored_text(text, Colors.UNDERLINE + Colors.MAGENTA)

def get_underlined_cyan_str(text) -> str:
    return colored_text(text, Colors.UNDERLINE + Colors.CYAN)

def get_underlined_white_str(text) -> str:
    return colored_text(text, Colors.UNDERLINE + Colors.WHITE)

# Functions for bold and underlined text in different colors
def get_bold_underlined_red_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.UNDERLINE + Colors.RED)

def get_bold_underlined_green_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.UNDERLINE + Colors.GREEN)

def get_bold_underlined_yellow_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.UNDERLINE + Colors.YELLOW)

def get_bold_underlined_blue_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.UNDERLINE + Colors.BLUE)

def get_bold_underlined_magenta_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.UNDERLINE + Colors.MAGENTA)

def get_bold_underlined_cyan_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.UNDERLINE + Colors.CYAN)

def get_bold_underlined_white_str(text) -> str:
    return colored_text(text, Colors.BOLD + Colors.UNDERLINE + Colors.WHITE)

def log_success(msg:str, showIcon:bool  = False):
    print(get_green_str(f'✔ {msg}') if showIcon else get_green_str(msg))

def perror(msg:str = "autogit: fatal error occured"):
    print(Colors.RED + msg + Colors.RESET)

def is_git_installed():
    try:
        # Run the 'git --version' command
        subprocess.run(['git', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        return False
    

def execsh(command:str):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)
    
def is_git_repo():
    abs_path = os.path.abspath("./.git")
    return os.path.exists(abs_path) and os.path.isdir(abs_path)

def append_stop(input_string, suffix):
    return input_string + suffix if input_string else ""

def print_progress_bar(current_value, total_value):   
    percentage = (current_value / total_value) * 100
    filled_length = int(percentage / 5)  # Assuming each dash represents 5% progress
    filled_length = min(filled_length, 20)  # Limit the filled length to 20 characters
    bar = '\033[93m [' + '█' * filled_length + ' ' * (20 - filled_length) + ']\033[0m '
    sys.stdout.write("\r" + bar)
    sys.stdout.write("\r\033[?25l" + bar + f'{Colors.BOLD + Colors.BLUE}Next push in {total_value - current_value} seconds remaining {Colors.RESET}')  # Disable cursor blinking and hide cursor
    sys.stdout.flush()