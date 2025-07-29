from ColBERT.colbert.infra import Run, RunConfig, ColBERTConfig
from ColBERT.colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="hotpotqa_wiki")):

        config = ColBERTConfig(
            nbits=2,
            root="/ColBERT-main/experiments",
        )
        indexer = Indexer(checkpoint="./colbert", config=config)
        indexer.index(name="hotpotqa_wiki.nbits=2", collection="enwiki-20171001-pages-meta-current-withlinks-abstracts_clean.tsv")