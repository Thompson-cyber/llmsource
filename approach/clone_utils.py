import os
import subprocess
from subprocess import Popen, PIPE, STDOUT

from sklearn.metrics.pairwise import cosine_similarity


def analysis_log(log):
    fingerprints = set()
    clone_dict = {}
    for i in range(0, len(log)):
        line = log[i]
        if 'duplicate lines with fingerprint' in line:
            # update the information
            suffix = line.split('fingerprint ')[1]
            fingerprint = suffix.split(' in')[0]
            prefix = line.split(' duplicate')[0]
            clone_line_len = int(prefix.split('Found ')[1])
            generated_py_flag = False
            a_may_clone_code_flag = False
            for found_line_num in range(i + 1, len(log)):
                found_line = log[found_line_num]
                if "Between lines " in found_line:
                    if "generated.py" in found_line:
                        generated_py_flag = True
                        generated_line = found_line
                    if "a_may_clone_code" in found_line:
                        a_may_clone_code_flag = True
                else:
                    break
            if generated_py_flag and a_may_clone_code_flag:
                fingerprints.add(fingerprint)
                if fingerprint in clone_dict:
                    clone_dict[fingerprint].add(generated_line)
                else:
                    temp_set = set()
                    temp_set.add(generated_line)
                    clone_dict[fingerprint] = temp_set
    # xxx duplicate lines with figerprint xxx.bwtew
    return fingerprints, clone_dict


def run_simian_clone(temp_folder,clone_type):
    java_executable = 'java'

    jar_file = 'simian-2.5.10.jar'
    if clone_type == "Type2":
      # command = [java_executable, '-jar', jar_file, temp_folder]
      command = [java_executable, '-jar', jar_file, '-ignoreIdentifiers', 'true', '-ignoreIdentifierCase', 'true',
                 '-ignoreStrings', 'true', '-ignoreStringCase', 'true', '-ignoreNumbers', 'true', '-ignoreCharacters',
                 'true', '-ignoreCharacterCase', 'true', '-ignoreLiterals', 'true', '-ignoreSubtypeNames', 'true',
                 '-ignoreVariableNames', 'true', temp_folder]
    elif clone_type == "Type1":
      command = [java_executable, '-jar', jar_file, temp_folder]
    try:
        p = Popen(command, stdout=PIPE, stderr=STDOUT)
        log = []
        for line in p.stdout:
            log.append(line.decode('utf-8'))

    except subprocess.CalledProcessError as e:
        pass
    return log


#from sentence_transformers import SentenceTransformer
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
#model = SentenceTransformer('mchochlov/codebert-base-cd-ft')


#def get_codebert_embeddings(folder_path):
#    embeddings = {}
#    for filename in os.listdir(folder_path):
#        file_path = os.path.join(folder_path, filename)
#        if os.path.isfile(file_path) and filename.endswith(".py"):
#            with open(file_path, "r", encoding="utf-8") as file:
#                code = file.read()
            # Extract embeddings
#            embeddings[filename] = model.encode(code)

#    return embeddings

#def graph_code_bert_clone(temp_path,generated_filename="generated.py",threshold=0.95):
#    embeddings = get_codebert_embeddings(temp_path)

#    generated_embedding = embeddings.get(generated_filename)

    # Compare with other embeddings
#    similar_file = None
#    max_similarity = -1.0
#    for filename, embedding in embeddings.items():
#        if filename != generated_filename:
#            similarity = cosine_similarity([generated_embedding], [embedding])[0][0]
#            return similarity
            # if similarity >= threshold and similarity > max_similarity:
            #     max_similarity = similarity
            #     similar_file = filename
            #     return similarity
    # return -1
    # return similar_file, max_similarity