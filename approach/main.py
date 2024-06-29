'''
Input: folder
Output: <generated_file, detected_clone>, <generated_file,detected_clone, {[source:clone_content...]}>
    Other Output: "statistics": #filter_file,#clone_file,#loc_clone, #time

Methods:
    def get_files(folder):

    def filter_file(code): # by line of code

    def get_comments(code):
    def get_signatures(code):

    def search_lucene(query,index_flag):

    def get_code_database(hash,database):

    def clone_simian(temp_code_folder): # analysis log
    def clone_DNN(temp_code_folder):

Workflow:
1: Input: model_name,generated_folder,log_folder, temp_path(code_clone),
2: get all files of given generated_folder
    3: for each file, filter file
    4: generate query by comments and signatures
    5: two manner detect clone
        quick: two file compare, find-> exist
        all: compare all files
        analysis clone log
    6: save clone
'''
import argparse
import datetime
import os
import sys

from tqdm import tqdm

from clone_utils import run_simian_clone, analysis_log
#from clone_utils import graph_code_bert_clone
from code_utils import filter_code, generate_queries,generate_full_queries
from file_utils import get_files, delete_files_in_folder
from log_utils import write_line
from search_utils import get_code_in_mongodb, search_by_lucene_cp, search_by_lucene_ts
from sklearn.metrics.pairwise import cosine_similarity

def run_clone(file, temp_path, log_path, h, license, database_flag):
    logs = run_simian_clone(temp_path + "/*.py","Type1")
    write_line(logs, log_path + "/clone_logs.txt")
    fingerprints, clone_dict = analysis_log(logs)
    if len(fingerprints) != 0:
        content = file + "," + database_flag + ",CodeHash:" + h + "," + str(fingerprints) + "," + str(clone_dict) + '\n'
        write_line(content, log_path + "/clone_full.txt")
        write_line(file, log_path + "/clone_file.txt")
        write_line(file + "," + h + "," + license, log_path + "/license.txt")
        return True
    return False

from sentence_transformers import SentenceTransformer
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model = SentenceTransformer('mchochlov/codebert-base-cd-ft')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def get_codes_clone(temp_path, log_path, generated_code, file, database_flag, hash_set1, hash_set2):
    list_set = list(hash_set1)
    list_set.extend(hash_set2)
    generated_embedding = model.encode(generated_code)
    for i in range(0, len(list_set)):
        # if CLONE_FLAG: break
        h = list_set[i]
        code, license = get_code_in_mongodb(database_flag, h)
        write_line(database_flag+","+h, log_path + "/searched_hash.txt")
        if code != None:
            # print(generated_code)
            # print(code)
            with open(temp_path + "/generated.py", 'a') as f:
                f.write(generated_code)
            with open(temp_path + "/a_may_clone_code" + str(i) + ".py", 'a') as f:
                f.write(code)
            # TODO:save the_stack
            if run_clone(file, temp_path, log_path, h, license, database_flag):
                print(file + h + code + ":Found")
                delete_files_in_folder(temp_path)
                return 
                
            else:
                #clone_emebdding = model.encode(code)
                #similarity = cosine_similarity([generated_embedding], [clone_emebdding])[0][0]
                #if similarity >= 0.90:
                #    content = file +","+ database_flag+",CodeHash:" + h + "," + str(similarity)
                #    write_line(content, log_path + "/clone_type_34.txt")
                
                #content = file +","+ database_flag+ ",CodeHash:" + h + "," + str(similarity)
                #write_line(content, log_path + "/similarity.txt")
                delete_files_in_folder(temp_path)
            #    delete_files_in_folder(temp_path)
            # else:
            #    similarity = graph_code_bert_clone(temp_path)
            #    if similarity >= 0.90:
            #        content = file +","+ database_flag+",CodeHash:" + h + "," + str(similarity)
            #        write_line(content, log_path + "/clone_type_34.txt")
            
            #    content = file +","+ database_flag+ ",CodeHash:" + h + "," + str(similarity)
            #    write_line(content, log_path + "/similarity.txt")
            #    delete_files_in_folder(temp_path)


# python main.py --generated_path /home/dl2/user_disk/generated_code/codeT5p-220M-py/instruct/ --log_path /home/dl2/user_disk/generated_code/codeT5p-220M-py/search_clone/0401 --temp_path temp_codet5p
def main():
    quick = True

    starttime = datetime.datetime.now()
    generated_path = args.generated_path
    log_path = args.log_path
    temp_path = args.temp_path

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    for file in tqdm(get_files(generated_path), desc="Processing Files"):
        # CLONE_FLAG = False
        with open(file, 'r') as f:
            generated_code = f.read()
        if filter_code(generated_code):
            write_line(file, log_path + "/filter_file.txt")
            continue
            
        all_queries = generate_full_queries(generated_code)
        search_ts_hash = search_by_lucene_ts(all_queries, "comments")
        search_cp_hash = search_by_lucene_cp(all_queries, "comments")
        search_ts_hash2 = search_by_lucene_ts(all_queries, "method_signature")
        search_cp_hash2 = search_by_lucene_cp(all_queries, "method_signature")
        #  search_by_lucene_cp(all_queries, "method_call")
        # search_by_lucene_cp(all_queries, "variables")
        #comments_queries, signature_queries = generate_queries(generated_code)
        
        #search_ts_hash = search_by_lucene_ts(comments_queries, "comments")
        #search_cp_hash = search_by_lucene_cp(comments_queries, "comments")
        #search_ts_hash2 = search_by_lucene_ts(signature_queries, "method_signature")
        #search_cp_hash2 = search_by_lucene_cp(signature_queries, "method_signature")
        
        get_codes_clone(temp_path, log_path, generated_code, file, "codeparrot", search_cp_hash, search_cp_hash2)
        get_codes_clone(temp_path, log_path, generated_code, file, "the_stack", search_ts_hash, search_ts_hash2)


    endtime = datetime.datetime.now()
    write_line(str(endtime - starttime), log_path + "/time.txt")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, required=True, help="The code generation model name")
    parser.add_argument('--generated_path', type=str, default="",
                        help="The path that save the generated code")
    parser.add_argument('--log_path', type=str, default="",
                        help="The path that save the log information")
    parser.add_argument('--temp_path', type=str, default="",
                        help="The path that used for code clone")
    return parser.parse_args(argv)


if __name__ == '__main__':
    # python generate.py --model "codegen-350M-mono" --start_id 1 --prompt "instruct" --save_path "/home/dl2/user_disk/generated_code/"
    args = parse_arguments(sys.argv[1:])
    main()
