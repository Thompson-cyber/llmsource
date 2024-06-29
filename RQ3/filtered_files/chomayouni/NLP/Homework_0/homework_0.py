# HW Zero is meant to get you familiar with the basics of NLP and Regex expressions.
# Written by: Connor Homayouni
# Assisted by github copilot

import re
import os
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import matplotlib.pyplot as plt
nltk.download('punkt')
# nltk.download('wordnet')

# for debugging
print(os.getcwd())

#This scrapes the absolute path of the file from the current working directory
# the Text file that your are reading must be in the same directory as the python file
#__file__ is the current file that is being run
working_directory = Path(__file__).absolute().parent

def Normalize(file_path, lower, stem, stop_word, search_phrase, find_characters):

    # lowercase, tokenize, remove stopwords 
    #-- the list of stopwards was grabbed from https://gist.github.com/sebleier/554280?permalink_comment_id=3431590#gistcomment-3431590
    #-- I removed some of the stopwords that I thought were important to the text
    mystops = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz",]
    mystops = set(mystops)
    custom_stops = ["ó", "A", "òa", "òit", "fin", 'int', 'ext'] # these are script specific stopwords that I found while reading the text
    custom_stops = set(custom_stops)
    stemmer = PorterStemmer() if stem else None
    phrase_count = 0
    character_counts = Counter()
    characters = ['Luke','Luke Skywalker','Leia','Princess Leia','Han Solo','Han','Darth Vader','Darth','Vader','Ben','Obi','Obi-Wan', "Kenobi",'Skywalker','Sith','Jedi','Chewbacca','Threepio', 'Artoo','Biggs','Jabba', 'Owen', 'Beru', 'Lars', 'Lando', 'Calrisian', 'Yoda', 'Emperor', 'Boba Fett', 'Palpatine', 'Wedge', 'Wedge Antilles', 'Tarkin', 'Tusken', 'Tusken Raider', 'Greedo', 'Red Leader', 'Red Two', 'Red Three', 'Red Four', 'Red Five' ]
    
    #read the file that was passed in
    with open(working_directory / file_path, 'r', encoding='utf-8') as file:
        text = file.read() 
    
    if lower:
        text = text.lower()
        search_phrase = search_phrase.lower() if search_phrase else None
        characters = [character.lower() for character in characters]

    if search_phrase:
        phrase_count = len(re.findall(search_phrase, text))

    if find_characters:
        for character in characters:
            character_counts[character] = len(re.findall(re.escape(character), text))

    tokens = word_tokenize(text)
    # print(text)
    
    if stop_word:
        tokens = [token for token in tokens if token not in mystops and token not in custom_stops]

    processed_tokens = []
    for token in tokens:
        if token.isalpha() and not (find_characters and token in characters):
            if stem:
                token = stemmer.stem(token)
            processed_tokens.append(token)
    
    # Trimming the non-alphanumeric characters
    token_counts = Counter(processed_tokens)
    
    return token_counts, phrase_count, character_counts if find_characters else Counter()

def plot_most_common_words(word_counts, num_words=100, ax=None):
    # Get the most common words and their counts

    if ax is None:
        ax = plt.gca()
        
    common_words = word_counts.most_common(num_words)
    words, counts = zip(*common_words)

    # Creating the line plot
    ax.plot(words, counts, color='skyblue', marker='o')
    ax.set_xlabel('Words')
    ax.set_ylabel('Counts')
    ax.set_title('Top 100 Most Common Words')
    ax.set_xticks(words)
    ax.set_xticklabels(words, rotation=90)  # Rotate word labels for readability
    ax.grid(True)
    # ax.set_xticklabels(words, rotation=45)


def plot_search_phrase_count(phrase_count, search_phrase, ax=None):
    # Create a bar plot for the search phrase count
    if ax is None:
        ax = plt.gca()

    # plt.figure(figsize=(5, 3))
    ax.bar(search_phrase, phrase_count, color='tomato')
    ax.set_xlabel('Phrase')
    ax.set_ylabel('Count')
    ax.set_title(f"Count of '{search_phrase}' in Text")

def plot_character_counts(character_counts, ax=None):
    if ax is None:
        ax = plt.gca()

    characters, counts = zip(*character_counts.items())
    ax.barh(characters, counts, color='green')
    ax.set_xlabel('Counts')
    ax.set_ylabel('Characters')
    ax.set_title('Character Mention Counts')
    ax.invert_yaxis()

def main():
    # initialize parser
    parser = argparse.ArgumentParser(  description="Normalize a text file and count the number of times each word appears.")
    print(parser.description)
    
    # arguments used when starting the program
    parser.add_argument("file_path", help="Path to the file you want to read.")
    parser.add_argument("-l", "--lower", help="Convert all words to lowercase.", action="store_true")
    parser.add_argument("-s", "--stem", help="Stem words.", action="store_true")
    parser.add_argument("-sw", "--stop_word", help="Remove all stopwords from the final text.", action="store_true")
    parser.add_argument("-ph",'--search_phrase', type=str, help="Search for a specific phrase in the text.")
    parser.add_argument("-ch", "--find_characters", help="Search for the total number of times the main characters are mentioned.", action="store_true")
    
    args = parser.parse_args()
    print('Below are the arguments that were passed in:')
    print(args)
    word_count, phrase_count, character_counts= Normalize(args.file_path, args.lower, args.stem, args.stop_word, args.search_phrase, args.find_characters)
    most_common_words = word_count.most_common(10)
    least_common_words = word_count.most_common()[:-11:-1]


    # Sort and print (this is for debugging)
    for word, count in word_count.most_common():
        print(f"{word} {count}")

    # Print the top 10 most and least common words
    print("\nTop 10 Most Common Words:")
    for word, count in most_common_words:
        print(f"{word}: {count}")

    print("\nTop 10 Least Common Words:")
    for word, count in least_common_words:
        print(f"{word}: {count}")

        
    
    print(f"\nNumber of times '{args.search_phrase}' appears in the text: {phrase_count}")
    total_tokens = sum(word_count.values())
    print(f"Total number of tokens: {total_tokens}")



    # Plotting the most and least common words
    plt.figure(figsize=(15, 5))
    # Plot for most common words
    ax1 = plt.subplot(1, 2, 1)
    words, counts = zip(*most_common_words)
    ax1.plot(words, counts, marker='o', color='blue')
    ax1.set_title('Top 10 Most Common Words')
    ax1.set_xlabel('Words')
    ax1.set_ylabel('Frequency')
    ax1.set_xticklabels(words, rotation=45)

    # Plot for least common words
    ax2 = plt.subplot(1, 2, 2)
    words, counts = zip(*least_common_words)
    ax2.plot(words, counts, marker='o', color='red')
    ax2.set_title('Top 10 Least Common Words')
    ax2.set_xlabel('Words')
    ax2.set_ylabel('Frequency')
    ax2.set_xticklabels(words, rotation=45)  

    num_plots = 1 + (1 if args.search_phrase else 0) + (1 if args.find_characters else 0)
    plt.figure(figsize=(15 * num_plots, 10))  # Adjust figure size based on the number of plots

    # This combines the two plots into one figure
    ax1 = plt.subplot(1, num_plots, 1)  
    plot_most_common_words(word_count, ax=ax1)

    current_subplot = 2
    if args.search_phrase:
        ax2 = plt.subplot(1, num_plots, current_subplot)
        plot_search_phrase_count(phrase_count, args.search_phrase, ax=ax2)
        current_subplot += 1

    if args.find_characters:
        ax3 = plt.subplot(1, num_plots, current_subplot)
        plot_character_counts(character_counts, ax=ax3)

    # Sort words by frequency and get the sorted frequencies - Generated by chatGPT
        sorted_word_counts = sorted(word_count.values(), reverse=True)
        average_frequency = np.mean(sorted_word_counts)
        plt.figure(figsize=(10, 5))
        ax_main = plt.subplot(1, 1, 1)
        ax_main.plot(sorted_word_counts, label='Word Frequencies', marker='o')
        ax_main.axhline(y=average_frequency, color='r', linestyle='--', label=f'Average Frequency: {average_frequency:.2f}')

        ax_main.set_title('Overall Word Frequency Distribution')
        ax_main.set_xlabel('Words (Ranked by Frequency)')
        ax_main.set_ylabel('Frequency (log scale)')
        ax_main.set_yscale('log')
        ax_main.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
