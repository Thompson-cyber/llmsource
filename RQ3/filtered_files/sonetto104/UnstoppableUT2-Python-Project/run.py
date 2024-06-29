import gspread
import os
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from rich.console import Console
import time
import colorama
import re
import datetime
import pandas as pd
from tabulate import tabulate
from termcolor import colored

# Load environment variables from .env file
load_dotenv()

# Define OAuth 2.0 scopes
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

# Load service account credentials

CREDS = Credentials.from_service_account_file("creds.json")
SCOPED_CREDS = CREDS.with_scopes(SCOPE)

# Authorize gspread with scoped credentials
GSPREAD_CLIENT = gspread.authorize(SCOPED_CREDS)

# Open Google Sheets documents
SHEET = GSPREAD_CLIENT.open('UnstoppableUT2 Tracker Spreadsheet')
USERNAME_PASSWORD_DATA_SHEET = GSPREAD_CLIENT.open('UnstoppableUT2 Username and Password Data Spreadsheet')

# Get email address from environment variables
email_address = os.getenv("EMAIL_ADDRESS")


colorama.init()

G = colorama.Fore.LIGHTGREEN_EX
R = colorama.Fore.RED
B = colorama.Fore.CYAN
Y = colorama.Fore.YELLOW
W = colorama.Fore.WHITE
M = colorama.Fore.MAGENTA

console = Console()


def print_banner():
    """
    Print welcome banner/logo for user
    """
    console.print('')
    console.print(' _   _                                                      _      _', style="color(200)")
    console.print('| | | |  ____       ___   _                                | |    | | _____ ', style="color(200)")
    console.print('| | | | /    \     / __|_| |_  _____  _____  _____  ______ | |____| ||  _  |', style="color(200)")
    console.print('| | | |/  /\  |   / /  |_   _||  _  ||  _  ||  _  ||  _   ||  _  || || |_| |', style="color(208)")
    console.print('| | | || |  | |   \ \    | |  | | | || | | || | | || | |  || | | || ||  ___|', style="color(208)")
    console.print('\ \_/ /| |  | | __/ /    | |_ | |_| || |_| || |_| || |_|  || |_| || || |___     ' , style="color(208)")
    console.print(' \___/ |_|  |_||___/     |__ ||_____||  ___|| ____||____|\||_____||_||_____|', style="color(208)")
    console.print('               __       __           | |    | |  ', style="color(200)")
    console.print('              |  |     |  |          |_|    |_|  ', style="color(200)")
    console.print('              |  |     |  | ____________ ', style="color(200)")
    console.print('              |  |     |  ||____    ____|  ___________ ', style="color(200)")
    console.print('              |  |     |  |     |  |      /  _______  \ ', style="color(199)")
    console.print('              |  |     |  |     |  |     /  /       \  \ ', style="color(199)")
    console.print('              |  |     |  |     |  |    |__/        /  /', style="color(199)")
    console.print('              |  |     |  |     |  |               /  /', style="color(199)")
    console.print('              |  |     |  |     |  |              /  /', style="color(197)")
    console.print('              |  |     |  |     |  |             /  /', style="color(197)")
    console.print('              \  \_____/  /     |  |            /  /', style="color(197)")
    console.print('               \_________/      |  |           /  /', style="color(197)")
    console.print('                                |  |          /  /', style="color(196)")
    console.print('                                |__|         /  /', style="color(196)")
    console.print('                                            /  /', style="color(196)")
    console.print('                                           /  /__________ ', style="color(196)")
    console.print('                                          |______________| ', style="color(196)")
    console.print('                                             ')


def print_incrementally(console, text):
    
    for char in text:
        console.print(char, end='', highlight=False)
        console.print('', end='', highlight=False)  # Flush the output
        time.sleep(0.02)
    console.print('')  # Add a newline after the text is printed


def new_user_or_existing_user():
    """
    Prompts the user to choose between new or existing user, and performs the corresponding actions.
    """
    console = Console()
    
    welcome_text = (
        "\n"
        "\n"
        f"{W}Welcome to Unstoppable UT2, where you can keep track of your UT2 performance.\n"
        "In case you're unfamiliar with the term 'UT2', it refers to an aerobic workout\n"
        "at an intensity which can be held for the full workout duration.\n"
        "You should be comfortable enough to speak and be operating at 65-75% maximum\n"
        "heart rate.\n"
        "The workout should last approximately 60 minutes.\n"
    )

    print_incrementally(console, welcome_text)

    while True:
        choice_text = f"{Y}Type 1 if you are a new user, or type 2 if you are an existing user: "
        print_incrementally(console, choice_text)
        
        choice = input()
        if choice == '1':
            register_new_user()
            break
        elif choice == '2':
            login_existing_user()
            break
        else:
            error_text = f"{R}Invalid choice. Please try again.\n"
            print_incrementally(console, error_text)


def register_new_user():
    """
    Registers a new user.
    """
    while True:
        username = type_username()
        if search_username(username):
            username_exists_text = f"{R}Username already exists. Please select a different one."
            print_incrementally(console, username_exists_text)
        else:
            password = type_new_password()
            write_username_and_password_to_data_sheet(username, password)
            search_file(username)
            break


def type_username():
    """
    Here is where the user will enter
    their username.
    """
    type_username_text = f"{Y}Please type your username below.\n{W}It must contain a minimum of five characters.\nIt must contain only lowercase letters, no spaces, no numbers and\nno special characters or symbols.\n"
    print_incrementally(console, type_username_text)
    while True:
        username = input(f"{Y}Please type your username here: ")
        if len(username) < 5:
            print("Username must contain a minimum of 5 characters.")
        elif not re.match("^[a-z]*$", username):
            print("Username must contain only lowercase letters without spaces, numbers or symbols.")
        else:
            return username


def search_username(username):
    """
    This function will check the username_password_data_sheet
    to see if the username already exists.
    """
    worksheet = USERNAME_PASSWORD_DATA_SHEET.sheet1
    usernames = worksheet.col_values(1)
    if username in usernames:
        return True
    else:
        return False


def type_new_password():
    """
    This is where the user will type their password.
    """
    new_password_text = f"{Y}Please type your password below.\n{W}It must contain a minimum of five characters.\nIt must contain only lowercase letters, no spaces, no numbers and\nno special characters or symbols.\n"
    print_incrementally(console, new_password_text)
    while True:
        password = input(f"{Y}Please type your password here: ")
        if len(password) < 5:
            print("Password must contain a minimum of 5 characters.")
        elif not re.match("^[a-z]*$", password):
            print("Password must contain only lowercase letters without spaces, numbers or symbols.")
        else:
            return password


# This code block was taken almost directly from
# the documentation for Google's Drive API.
# You can find it here: https://developers.google.com/drive/api/guides/search-files
def search_file(username):
    """Search file in drive location

    Load pre-authorized user credentials from the environment.
    """
    creds = CREDS

    try:
        # create drive api client
        service = build('drive', 'v3', credentials=creds)
        files = []
        page_token = None
        while True:
            # pylint: disable=maybe-no-member
            response = service.files().list(q=f"name='{username} UT2 Tracker Spreadsheet'",
                                            spaces='drive',
                                            fields='nextPageToken,'
                                                   'files(name)',
                                            pageToken=page_token).execute()
            for file in response.get('files', []):
                # Process change
                if f'{username} UT2 Tracker Spreadsheet' in file['name']:
                    existing_user_choice(username)
                    return
            files.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if not files:
                print(f"{G}Thanks for signing up {M}{username}!\n")
                create_new_user_workbook(username)
                user_workout_choice(username)
            if page_token is None:
                break       
    except HttpError as error:
        print(F'An error occurred: {error}')
        files = None

    return files


def existing_user_choice(username):
    """
    This will allow an existing user
    to decide if they want to log a new workout
    or get data about their previous workouts.
    """
    welcome_back_text = f"{W}Welcome back {M}{username}!{W} What would you like to do today?\n{B}1. Log a new workout\n2. View the data from previous workouts\n3. View your averge scores from your last three workouts\n"
    print_incrementally(console, welcome_back_text)
    user_choice = None

    while user_choice not in ['1', '2', '3',]:
        user_choice = input(f"{Y}Type 1, 2 or 3 to choose one of the above: ")
        print("")
    user_choice = int(user_choice)
    
    if user_choice == 1:
        print("You've chosen to log a new workout.\n")
        user_workout_choice(username)
    if user_choice == 2:
        print("You've chosen to view the data from your previous workouts.\n")
        #  Add style and incremental printing here
        print("Type 1 to view your treadmill workout data.\nType 2 to view your rowing ergometer data.\nType 3 to view your exercise bike data.")
        worksheet = None
        while worksheet not in ['1', '2', '3']:
            worksheet = input("Type 1, 2 or 3 to choose one of the above: ")
            print("")
        worksheet = int(worksheet)
        if worksheet == 1:
            worksheet = "Treadmill"
        elif worksheet == 2:
            worksheet = "Rowing Ergometer"
        elif worksheet == 3:
            worksheet = "Exercise Bike"
        display_all_previous_workout_entries(worksheet, username)

    if user_choice == 3:
        print("You've chosen to view your averge scores from your last three workouts.")
        print("Type 1 to view your average treadmill workout data.\nType 2 to view your average rowing ergometer data.\nType 3 to view your average exercise bike data.")
        worksheet = None
        while worksheet not in ['1', '2', '3']:
            worksheet = input("Type 1, 2 or 3 to choose one of the above: ")
            print("")
        worksheet = int(worksheet)
        if worksheet == 1:
            worksheet = "Treadmill"
        elif worksheet == 2:
            worksheet = "Rowing Ergometer"
        elif worksheet == 3:
            worksheet = "Exercise Bike"
        calculate_average_workout_scores(worksheet, username)
 

def login_existing_user():
    """
    Logs in an existing user.
    """
    while True:
        username = type_username()
        if search_username(username):
            password = type_new_password()
            if check_password(username, password):
                existing_user_choice(username)
                break
            else:
                incorrect_password_text = f"{R}Incorrect password. Please try again."
                print_incrementally(console, incorrect_password_text)
        else:
            username_not_found_text = f"{R}Username not found."
            print_incrementally(console, username_not_found_text)
            response = input(f"{Y}Type 1 to enter a new username or press Enter to try again: ")
            if response == "1":
                new_user_or_existing_user()

         
def existing_user():
    """
    This function is called if the user is an existing user.
    """
    while True:
        username = type_username()
        if search_username(username):
            password = type_new_password()
            if check_password(username, password):
                existing_user_choice(username)
                return
            else:
                incorrect_password_text = f"{R}Incorrect password. Please try again."
                print_incrementally(console, incorrect_password_text)
        else:
            username_not_found_text = f"{R}Username not found. Please try again."
            print_incrementally(console, username_not_found_text)


def check_password(username, password):
    """
    This function checks if the given password matches the password for the given username.
    """
    worksheet = USERNAME_PASSWORD_DATA_SHEET.sheet1
    row = None
    try:
        cell = worksheet.find(username)
        row = cell.row
        stored_password = worksheet.cell(row, 2).value
    except gspread.exceptions.CellNotFound:
        return False
    if stored_password == password:
        return True
    else:
        return False


def write_username_and_password_to_data_sheet(username, password):
    """
    This will add the user's username and password
    to the username and password spreadsheet.
    """
    worksheet = USERNAME_PASSWORD_DATA_SHEET.sheet1
    next_row = len(worksheet.get_all_values()) + 1
    new_row = [username, password]
    worksheet.insert_row(new_row, next_row)
    print(f"{G}User added successfully!\n")


def create_new_user_workbook(username):
    """
    Creates new spreadsheet in Google Sheets.
    """
    # Create new workbook with three worksheets.
    # Workbook will include user's typed username.
    # Three worksheets have titles listed below.
    # Default "Sheet1" worksheet gets deleted.
    # Worksheet list object created to be referred to later when formatting.
    user_workbook = GSPREAD_CLIENT.create(f"{username} UT2 Tracker Spreadsheet")
    worksheet_names = ["Treadmill", "Rowing Ergometer", "Exercise Bike"]
    worksheets = []
    for worksheet_name in worksheet_names:
        worksheet = user_workbook.add_worksheet(title=worksheet_name, rows=1000, cols=3)
        worksheets.append(worksheet)
    user_workbook.del_worksheet(user_workbook.sheet1)
    # Add cell formatting rule to be applied to all worksheets.
    cell_format = {
        "textFormat": {
            "bold": True
        }
    }
    cells_to_format = ['A1', 'B1', 'C1']
    for cell in cells_to_format:
        for worksheet in worksheets:
            worksheet.format(cell, cell_format)
    # Created nested for loops to iterate cell headings
    # across all three of user's worksheets.
    all_worksheet_types = user_workbook.worksheets()
    worksheet_headings = ["Date", "Duration", "Distance"]
    for i, worksheet in enumerate(all_worksheet_types):
        for j in range (3):
            worksheet.update_cell(1, j+1, worksheet_headings[j])
    if email_address:
        user_workbook.share(email_address, perm_type='user', role='writer')
    else:
        print("Email address not found in the environment variables.")


def user_workout_choice(username):
    """
    This will allow user to select which workout they want to log
    and write their data to the appropriate worksheet.
    """
    workout_choice_text = f"What kind of workout would you like to log today?\n{M}1. Treadmill\n2. Rowing Ergometer\n3. Exercise Bike"
    print_incrementally(console, workout_choice_text)

    workout_choice = None

    while workout_choice not in ['1', '2', '3']:
        workout_choice = input(f"{Y}Type 1, 2 or 3 to choose one of the above.")
    workout_choice = int(workout_choice)
    print("")
    if workout_choice == 1:
        print(f"{M}You've chosen to update your treadmill data.")
        while True:
            time_data = input_workout_duration_info()
            if validate_user_workout_duration_input(time_data):
                break
        while True:
            distance_data = input_workout_distance_info()
            if validate_user_workout_distance_input(distance_data):
                break
        update_worksheet(time_data, distance_data, "Treadmill", username)

    elif workout_choice == 2:
        print(f"{M}You've chosen to update your rowing ergometer data.")
        while True:
            time_data = input_workout_duration_info()
            if validate_user_workout_duration_input(time_data):
                break
        while True:
            distance_data = input_workout_distance_info()
            if validate_user_workout_distance_input(distance_data):
                break
        update_worksheet(time_data, distance_data, "Rowing Ergometer", username)

    elif workout_choice == 3:
        print(f"{M}You've chosen to update your exercise bike data.")
        while True:
            time_data = input_workout_duration_info()
            if validate_user_workout_duration_input(time_data):
                break
        while True:
            distance_data = input_workout_distance_info()
            if validate_user_workout_distance_input(distance_data):
                break
        update_worksheet(time_data, distance_data, "Exercise Bike", username)


def display_all_previous_workout_entries(worksheet, username):
    """
    This function will allow the user to see their data from 
    past workouts.
    """
    username_sheet = GSPREAD_CLIENT.open(f'{username} UT2 Tracker Spreadsheet')
    workout_type_to_be_displayed = username_sheet.worksheet(worksheet)
    columns = []
    for ind in range(1, 4):
        column = workout_type_to_be_displayed.col_values(ind)
        columns.append(column)
    df = pd.DataFrame(columns).transpose()
    df.columns = ["Column 1", "Column 2", "Column 3"]

    # Convert DataFrame to tabulate table
    table = tabulate(df, headers='keys', tablefmt='grid')

    # Apply color to the entire table
    colored_table = colored(table, 'white', 'on_magenta')

    # Print the colorized table
    print(colored_table)

    return df


def calculate_average_workout_scores(worksheet, username):
    """
    This function will display the user's average
    workout duration and distance covered for a given
    workout type using the 3 most recent entries.
    """
    username_sheet = GSPREAD_CLIENT.open(f'{username} UT2 Tracker Spreadsheet')
    workout_type_to_be_displayed = username_sheet.worksheet(worksheet)
    duration_column_entries = workout_type_to_be_displayed.col_values(2)
    if len(duration_column_entries) >= 4:
        duration_column_last_three_entries = duration_column_entries[-3:]
        # Convert time values to seconds
        # The code on the line below used to convert existing time data in the worksheet into seconds was generated by ChatGPT.
        seconds = [datetime.datetime.strptime(t, '%H:%M:%S').time().second + datetime.datetime.strptime(t, '%H:%M:%S').time().minute * 60 + datetime.datetime.strptime(t, '%H:%M:%S').time().hour * 3600 for t in duration_column_last_three_entries]
        # Calculate average of seconds
        avg_seconds = sum(seconds) / len(seconds)
        # Convert average seconds back to hh:mm:ss format
        avg_time = str(datetime.timedelta(seconds=avg_seconds))
        print(f"Your average workout duration for your last three {M}{worksheet}")
        print(f"{W}workouts is {M}{avg_time}.")
    elif len(duration_column_entries) < 4 and len(duration_column_entries) > 1:
        print(f"{W}You haven't logged three {worksheet} workouts yet,")
        print(f"but here's your existing data anyway!")
        # Convert time values to seconds
        seconds = [datetime.datetime.strptime(t, '%H:%M:%S').time().second + datetime.datetime.strptime(t, '%H:%M:%S').time().minute * 60 + datetime.datetime.strptime(t, '%H:%M:%S').time().hour * 3600 for t in duration_column_entries[1:]]
        # Calculate average of seconds
        avg_seconds = sum(seconds) / len(seconds)
        # Convert average seconds back to hh:mm:ss format
        avg_time = str(datetime.timedelta(seconds=avg_seconds))
        print(f"Your average workout duration for your {M}{worksheet}")
        print(f"{W} is {M}{avg_time}.")
    elif len(duration_column_entries) <= 1:
        print(f"{W}You haven't logged any {worksheet} workouts yet.")

    distance_column_entries = workout_type_to_be_displayed.col_values(3)
    if len(distance_column_entries) >= 4:
        distance_column_last_three_entries = distance_column_entries[-3:]
        distance_column_last_three_entries = [float(entry) for entry in distance_column_entries[1:]]
        avg_distance = sum(distance_column_last_three_entries) / len(distance_column_last_three_entries)
        print(f"Your average distance covered in km for your last three")
        print(f"{M}{worksheet}{W} workouts is {M}{avg_distance}.")
    elif len(distance_column_entries) < 4 and len(distance_column_entries) > 1:
        distance_column_entries = [float(entry) for entry in distance_column_entries[1:]]
        avg_distance = sum(distance_column_entries) / len(distance_column_entries)
        print(f"{W}Your average distance covered in km for your last three")
        print(f"{M}{worksheet}{W} workouts is{M} {avg_distance}.")
    elif len(distance_column_entries) <= 1:
        pass


# The try/except format of this code block comes from
# Code Institute's Love Sandwiches Walkthrough Project.
def validate_user_workout_duration_input(time_data):
    """
    This will ensure that the user may only
    input data in this format - 00:00:00 -
    where the first two digits correspond to hours,
    the second two digits correspond to minutes
    and the last two digist correspond to seconds.
    """
    time_format = re.compile(r'^([01]\d|2[0-3]):([0-5]\d):([0-5]\d)$')
    try:
        match = time_format.fullmatch(time_data)
        if match is None:
            print(
                f"{R}Your workout time must be less than 24 hours.{W}The value for minutes must be\nless than 60. The value for seconds must be less than 60. You entered {R}{time_data}"
                )
            raise ValueError(
                f"{R}Your workout time must be less than 24 hours.\n{W}The value for minutes must be less than 60. The value for seconds must be less than 60.\nYou entered {R}{time_data}"
                )
        else:
            return True
    except ValueError as e:
        print(f"{R}Invalid data: {e}, please try again.\n")
        return False


# The try/except format for this code block was taken from
# Code Institue's Love Sandwiches Walkthrough Project.
def validate_user_workout_distance_input(distance_data):
    """
    This will ensure that the user may only
    input data in this format - 00.00 -
    the digits correspond to kilometers measured
    to two decimal places.
    """
    distance_format = re.compile(r'\d\d.\d\d')
    while True:
        try:
            distance_data_str = str(distance_data)
            match = distance_format.fullmatch(distance_data_str)
            if match is None:
                raise ValueError(
                    f"{R}Your distance in kilometres should be entered in this format - 00.00.\n You entered {distance_data}"
                    )
            break
        except ValueError as e:
            print(f"{R}Invalid data: {e}, please try again.\n")
            distance_data = input("Input your distance covered in kilometres in this format - 00.00 ")
    
    return True


def input_workout_duration_info():
    """
    Here the user will input their
    data for their workout duration.
    """
    while True:
        input_duration_text = f"{Y}Input your workout duration below\n{W}Your time should be entered in this format - 00:00:00\nE.g. if your workout was an hour and twenty minutes long, you would enter\n01:20:00.\nYour value for hours must be less than 24. Your value for minutes must be less than 60. Your value for seconds must be less than 60.\n"
        print_incrementally(console, input_duration_text)
        time_data = input(f"{Y}Please input your workout duration here: ")
        if time_data:
            break
    return time_data
        

def input_workout_distance_info():
    """
    Here the user will input their
    data for the distance covered in their workout.
    """
    while True:
        print("Input your distance covered in kilometres below.\n")
        print("Your distance should be entered in this format - 00.00\n")
        print("E.g. if you cycled 23.4km on the exercise bike, ")
        print("you would enter 23.40\n")
        distance_info_text = f"{Y}Input your distance covered in kilometres below.\n{W}Your distance should be entered in this format - 00.00\nE.g. if you cycled 23.4km on the exercise bike, you would enter 23.40\n"
        print_incrementally(console, distance_info_text)
        distance_data = input(f"{Y}Please input your workout distance here: ")
        if distance_data:
            print(f"{G}Thank you! Your UT2 Tracker data is being updated.")
            break
    return distance_data


# The basis for this code is taken almost directly from
# Code Institute's Love Sandwiches Walkthrough Project.
def update_worksheet(time_data, distance_data, worksheet, username):
    """
    This function will add the user's workout distance
    and duration data and append it to a row in their
    spreadsheet along with the date of data entry.
    """
    username_sheet = GSPREAD_CLIENT.open(f'{username} UT2 Tracker Spreadsheet')
    worksheet_to_update = username_sheet.worksheet(worksheet)
    current_date = datetime.datetime.now()
    date_string = current_date.strftime("%d-%m-%Y")
    row_to_append = [date_string, time_data, distance_data]
    worksheet_to_update.append_row(row_to_append)
    print(f"{G}{worksheet} worksheet updated successfully.")


def main():
    """
    Run all programme functions
    """
    while True:
        new_user_or_existing_user()
        continue_or_quit_choice = input(f"{Y}Type 1 to run the program again or 2 to leave Unstoppable UT2 for today: ")
        print("")
        if continue_or_quit_choice == '1':
            continue
        elif continue_or_quit_choice == '2':
            break
        else:
            print(f'{R}Invalid choice. Please try again.\n')


print_banner()
main()