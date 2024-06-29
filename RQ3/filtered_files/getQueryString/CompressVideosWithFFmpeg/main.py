# This code was generated with ChatGPT and is intended to improve the handling of ffmpeg.
import os
import subprocess

"""
input_dir = ""

output_dir = ""
"""


# Eigene Idee & Hilfe durch ChatGPT
def queue():
    year = 2021
    month = 6
    directory = 6

    while year != 2023 and month != 13:
        # Pfad zu den Videos, die komprimiert werden sollen
        input_dir = os.path.join("C:\\", "Users", "USER", "Desktop", "compressed", "nicht komprimiert", str(year),
                                 "{:02d}".format(month), "{:02d}_Videos".format(directory))

        # Pfad zum Speicherort der komprimierten Videos
        output_dir = os.path.join("C:\\", "Users", "USER", "Desktop", "compressed", "komprimiert", str(year),
                                  "{:02d}".format(month), "{:02d}_Videos".format(directory))

        print(input_dir)
        print(output_dir)
        print()

        month += 1
        directory += 1

        if month > 12:
            year += 1
            month = 1
            directory = 1

        compress(input_dir, output_dir)


def compress(input_dir, output_dir):
    # Überprüfen, ob das Eingabeverzeichnis existiert
    if not os.path.exists(input_dir):
        print("Eingabeverzeichnis nicht gefunden.")
        return

    # Überprüfen, ob das Ausgabeverzeichnis existiert. Wenn nicht, erstellen
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Anzahl der komprimierten Videos
    count = 0

    # Alle Videos im Eingabeverzeichnis durchlaufen
    for file in os.listdir(input_dir):
        # Überprüfen, ob die Datei eine Video-Datei ist, die von ffmpeg unterstützt wird
        if file.endswith((".mp4", ".avi", ".mkv", ".mov", ".wmv")):
            # Den vollständigen Pfad zum Eingabevideo erstellen
            input_path = os.path.join(input_dir, file)
            # Den vollständigen Pfad zum Ausgabevideo erstellen
            output_path = os.path.join(output_dir, file)
            # ffmpeg-Befehl ausführen, um das Video zu komprimieren und die Metadaten zu beibehalten
            ret_code = subprocess.run(
                ['ffmpeg', '-i', input_path, '-c:v', 'libx264', '-crf', '22', '-c:a', 'copy', '-vf', 'scale=-2:720',
                 '-map_metadata', '0', '-y', output_path], stderr=subprocess.PIPE)
            if ret_code == 0:
                count += 1
            else:
                print("Fehler bei der Komprimierung von " + file + ":")
                print(ret_code.stderr.decode())
        else:
            print("Videoformat nicht unterstützt: " + file)

    if count > 0:
        print(str(count) + " Video(s) komprimiert und im Verzeichnis " + output_dir + " gespeichert.")
    else:
        print("Keine Videos zum Komprimieren gefunden.")


queue()
# compress()
