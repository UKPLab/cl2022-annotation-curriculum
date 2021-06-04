import logging

import wget

from ca.paths import *
from ca.util import setup_logging


def _download_file(url: str, target_path: Path):
    import ssl

    if target_path.exists():
        logging.info("File already exists: [%s]", str(target_path.resolve()))
        return

    wget.download(url, str(target_path.resolve()))


def download_conll_2003():
    root = "https://raw.githubusercontent.com/patverga/torch-ner-nlp-from-scratch/master/data/conll2003/"
    url_train = root + "eng.train"
    url_dev = root + "eng.testa"
    url_test = root + "eng.testb"

    _download_file(url_train, PATH_IOB_CONLL_2003_TRAIN)
    _download_file(url_dev, PATH_IOB_CONLL_2003_DEV)
    _download_file(url_test, PATH_IOB_CONLL_2003_TEST)


def download_wnut17():
    root = "http://noisy-text.github.io/2017/files/"
    url_train = root + "wnut17train.conll"
    url_dev = root + "emerging.dev.conll"
    url_test = root + "emerging.test.annotated"

    _download_file(url_train, PATH_DATA_WNUT17_TRAIN)
    _download_file(url_dev, PATH_DATA_WNUT17_DEV)
    _download_file(url_test, PATH_DATA_WNUT17_TEST)


def download_spec():
    # http://pages.cs.wisc.edu/~bsettles/data/
    root = "http://pages.cs.wisc.edu/~bsettles/data/spec.tar.gz"
    _download_file(root, PATH_DATA_SPEC_ARCHIVE)


def download_sig_ie():
    root = "http://pages.cs.wisc.edu/~bsettles/data/sigie.tar.gz"
    _download_file(root, PATH_DATA_SIG_IE_ARCHIVE)


def download_habernal_convincing_arguments():
    root = "https://github.com/UKPLab/acl2016-convincing-arguments/archive/master.zip"
    _download_file(root, PATH_DATA_CONVINCING_ARCHIVE)


def main():
    setup_logging()
    PATH_DATA_EXTERN.mkdir(parents=True, exist_ok=True)
    PATH_DATA_EXTERN_CONLL_2003.mkdir(parents=True, exist_ok=True)
    PATH_DATA_WNUT17.mkdir(parents=True, exist_ok=True)

    download_conll_2003()
    download_wnut17()
    download_spec()
    download_sig_ie()
    download_habernal_convincing_arguments()


if __name__ == "__main__":
    main()
