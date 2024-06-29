import os


def get_files(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths[:10000]


def get_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for i in dirs:
            file_paths.append(folder_path + "\\" + str(i))
    return file_paths


def delete_files_in_folder(folder_path):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
            elif os.path.isdir(file_path):
                # If it's a directory, recursively call the function to delete its contents
                delete_files_in_folder(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")