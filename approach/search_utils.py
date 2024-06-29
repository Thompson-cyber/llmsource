import lucene
import pymongo

from java.io import File
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import IndexReader
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory, NIOFSDirectory
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version

lucene.initVM()

thestack_index_dir = "/home/dl2/disk/lucene/the_stack_index"
thestack_analyzer = StandardAnalyzer()
thestack_reader = DirectoryReader.open(NIOFSDirectory(Paths.get(thestack_index_dir)))
thestack_searcher = IndexSearcher(thestack_reader)

codeparrot_index_dir = "/home/dl2/user_disk/lucene/codeparrot"
codeparrot_analyzer = StandardAnalyzer()
codeparrot_reader = DirectoryReader.open(NIOFSDirectory(Paths.get(codeparrot_index_dir)))
codeparrot_searcher = IndexSearcher(codeparrot_reader)
client = pymongo.MongoClient("mongodb://localhost:27017/")
cp_db = client["Codeparrot_Python"]

cp_collection = cp_db["original_code"]


ts_db = client["The_Stack"]

ts_collection = ts_db["original_code"]


def search_by_lucene_cp(queries, column, number=20):
    cp_hash = set()
    for query in queries:
        try:
            query_cp = QueryParser(column, codeparrot_analyzer).parse(query)
            hits = codeparrot_searcher.search(query_cp, number)
            if hits.scoreDocs is not None and len(hits.scoreDocs) > 0:
                for scoreDoc in hits.scoreDocs:
                    doc = codeparrot_searcher.doc(scoreDoc.doc)
                    cp_hash.add(doc.get("hash"))
        except:
            continue
    return cp_hash


def search_by_lucene_ts(queries, column, number=20):
    ts_hash = set()
    for query in queries:
        try:
            query_ts = QueryParser(column, thestack_analyzer).parse(query)
            
            hits_ts = thestack_searcher.search(query_ts, number)
            
            if hits_ts.scoreDocs is not None and len(hits_ts.scoreDocs) > 0:
                for scoreDoc in hits_ts.scoreDocs:
                    
                    doc = thestack_searcher.doc(scoreDoc.doc)
                    
                    ts_hash.add(doc.get("hash"))
        except:
            continue
    return ts_hash


def get_code_in_mongodb(db_flag, hash):
    if db_flag == "codeparrot":
        result = cp_collection.find_one({"hash": hash})
        if result:
            code = result.get("code", "")
            license = result.get("license", "")
            return code, license
    if db_flag == "the_stack":
        result = ts_collection.find_one({"hash": hash})
        if result:
            code = result.get("code", "")
            license = result.get("license", "")
            return code, license
    return None, None
