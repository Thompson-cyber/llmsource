def write_line(content,path):
    with open(path,'a') as f:
        f.write(str(content)+"\n")