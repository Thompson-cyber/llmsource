import re
import tokenize
from io import BytesIO


def filter_code(code):
    if len(code.split('\n')) < 6:
        return True
    return False


def extract_comments(input_code):
    comments = []

    # Single-line comments
    single_line_pattern = r'#.*'
    single_line_matches = re.findall(single_line_pattern, input_code)
    comments.extend(single_line_matches)

    # Multi-line comments
    multi_line_pattern = r'(\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\")'
    multi_line_matches = re.findall(multi_line_pattern, input_code, re.DOTALL)
    for match in multi_line_matches:
        comments.append(match[0])

    return comments


def seperate_string_number(string):
    previous_character = string[0]
    groups = []
    newword = string[0]
    for x, i in enumerate(string[1:]):
        if i.isalpha() and previous_character.isalpha():
            newword += i
        elif i.isnumeric() and previous_character.isnumeric():
            newword += i
        else:
            groups.append(newword)
            newword = i

        previous_character = i

        if x == len(string) - 2:
            groups.append(newword)
            newword = ''
    return groups


def split_string(s):
    if "_" in s:
        s = " _ ".join(t for t in s.split("_"))
    upper = re.findall('[A-Z][^A-Z]*', s)
    if len(upper) != 0:
        s = " ".join(t for t in upper)
    digits = seperate_string_number(s)
    if len(digits) != 0:
        s = " ".join(t for t in digits)
    return s


def tokenize_method_signature(signature):
    # Convert the signature string to bytes
    signature_bytes = signature.encode('utf-8')

    # Use BytesIO to create a file-like object
    signature_file = BytesIO(signature_bytes)

    # Tokenize the method signature
    try:
      for token in tokenize.tokenize(signature_file.readline):
          if token.string=="utf-8" or token.string == "":
              continue
          tokens.append(split_string(token.string))
  
      return ' '.join(token for token in tokens if token.strip())
    except:
      return ''


def extract_signatures(code):
    pattern = r'def ([^\W\d]+\w*)\(([^)]*)\)'
    matches = re.findall(pattern, code)
    res = []
    for match in matches:
        function_name, parameters = match
        res.append(tokenize_method_signature(function_name + " " + parameters))
    return res

def generate_full_queries(code):
    all_queries = set()
    for line in code.split("\n"):
        if len(line) > 30000 or len(line) < 5:
            continue
        line = line.replace(":","")
        all_queries.add(line.lower())
    return all_queries

def generate_queries(code):
    comments_queries = set()
    signature_queries = set()
    comments = extract_comments(code)
    signatures = extract_signatures(code)
    for c in comments:
        if c == "#generate python code":
            continue
        if len(c) > 30000 or len(c) < 5:
            continue
        c = c.replace(":", "")
        comments_queries.add(c.lower())
    # method signature as query
    for m in signatures:
        if len(m) > 30000 or len(m) < 5:
            continue
        signature_queries.add(tokenize_method_signature(m))
    return comments_queries, signature_queries
