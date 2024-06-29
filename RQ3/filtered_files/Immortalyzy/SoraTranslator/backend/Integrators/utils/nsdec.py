""" this is a python implementation of nssdec.exe, generated by chatgpt
    the original code in C is at https://github.com/GoldbarGames/ONScripter-EN-Steam/blob/master/tools/nscdec.cpp
"""
import sys


def decode_ns_file(in_filename, out_filename="result.txt"):
    """decode a nscript.dat file to a result.txt file"""
    # File Opening
    try:
        in_fp = open(in_filename, "rb") if in_filename != "-" else sys.stdin.buffer
    except IOError:
        print(f"Couldn't open '{in_filename}' for reading")
        return False

    try:
        out_fp = open(out_filename, "wb") if out_filename != "-" else sys.stdout.buffer
    except IOError:
        print(f"Couldn't open '{out_filename}' for writing")
        return False

    try:
        # Decoding and Writing
        last_ch = b"\x00"
        ch = in_fp.read(1)
        while ch:
            decoded_ch = bytes([ch[0] ^ 0x84])
            if decoded_ch in (b"\r", b"\n") and last_ch != b"\r":
                out_fp.write(b"\n")
            else:
                out_fp.write(decoded_ch)
            last_ch = decoded_ch
            ch = in_fp.read(1)

        # Adding an ending newline if needed
        if last_ch not in (b"\r", b"\n"):
            out_fp.write(b"\n")

        # File Closing
        if in_filename != "-":
            in_fp.close()
        if out_filename != "-":
            out_fp.close()
        return True
    except:
        return False
