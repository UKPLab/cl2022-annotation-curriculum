import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from tarfile import TarFile
from zipfile import ZipFile

from ca.datasets import my_clean
from ca.paths import *


def conll2003_iob_to_bio(path_in: Path, path_out: Path):
    """
    Converts the IOB encoding from CoNLL 2003 to BIO encoding
    """

    with path_in.open() as f_in, path_out.open("w") as f_out:
        for line in f_in:
            if line.startswith("-DOCSTART-"):
                lastChunk = "O"
                lastNER = "O"
                f_out.write(line)
                continue

            if len(line.strip()) == 0:
                lastChunk = "O"
                lastNER = "O"
                f_out.write("\n")
                continue

            splits = line.strip().split()

            chunk = splits[2]
            ner = splits[3]

            if chunk[0] == "I":
                if chunk[1:] != lastChunk[1:]:
                    chunk = "B" + chunk[1:]

            if ner[0] == "I":
                if ner[1:] != lastNER[1:]:
                    ner = "B" + ner[1:]

            splits[2] = chunk
            splits[3] = ner

            f_out.write("\t".join(splits))
            f_out.write("\n")

            lastChunk = chunk
            lastNER = ner


def prepare_conll2003():
    PATH_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    PATH_DATA_PROCESSED_CONLL_2003.mkdir(parents=True, exist_ok=True)

    conll2003_iob_to_bio(PATH_IOB_CONLL_2003_TRAIN, PATH_DATA_CONLL_2003_TRAIN)
    conll2003_iob_to_bio(PATH_IOB_CONLL_2003_DEV, PATH_DATA_CONLL_2003_DEV)
    conll2003_iob_to_bio(PATH_IOB_CONLL_2003_TEST, PATH_DATA_CONLL_2003_TEST)


def prepare_spec():
    PATH_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with TarFile.open(PATH_DATA_SPEC_ARCHIVE) as tar:
        tar.extractall(path=PATH_DATA_PROCESSED)


def prepare_sig_ie():
    PATH_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with TarFile.open(PATH_DATA_SIG_IE_ARCHIVE) as tar:
        tar.extractall(path=PATH_DATA_PROCESSED)


def prepare_download_habernal_convincing_arguments():
    PATH_DATA_CONVINCING_ROOT.mkdir(parents=True, exist_ok=True)

    with ZipFile(PATH_DATA_CONVINCING_ARCHIVE) as myzip:
        for name in myzip.namelist():
            if name.startswith("acl2016-convincing-arguments-master/data/UKPConvArg1Strict-XML/") and name.endswith(
                ".xml"
            ):
                myzip.extract(name, PATH_DATA_CONVINCING_ROOT)


if __name__ == "__main__":
    prepare_conll2003()
    prepare_spec()
    prepare_sig_ie()
    prepare_download_habernal_convincing_arguments()
