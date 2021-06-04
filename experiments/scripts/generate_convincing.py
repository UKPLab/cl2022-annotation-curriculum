import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from tarfile import TarFile
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import stats

from ca.datasets import my_clean
from ca.paths import *


def load_convincing():
    document_ids = []
    args1 = []
    args2 = []
    labels = []
    t = []
    annotators = []

    for p in PATH_DATA_CONVINCING.iterdir():
        tree = ET.parse(p)
        root = tree.getroot()
        fmt = "%Y-%m-%d %H:%M:%S.%f UTC"

        for annotation in root:
            document_id = annotation.find("id").text
            for assignment in annotation.find("mTurkAssignments"):
                begin = assignment.find("assignmentAcceptTime")
                end = assignment.find("assignmentSubmitTime")

                begin = datetime.strptime(begin.text, fmt)
                end = datetime.strptime(end.text, fmt)
                time = (end - begin).total_seconds()

                label = assignment.find("value").text
                annotator = assignment.find("turkID").text

                score = float(assignment.find("turkCompetence").text)

                # if score < 0.7: continue

                arg1 = annotation.find("arg1").find("text").text
                arg2 = annotation.find("arg2").find("text").text

                document_ids.append(document_id)
                args1.append(my_clean(arg1))
                args2.append(my_clean(arg2))
                labels.append(label)
                t.append(time)
                annotators.append(annotator)

    data = {"document_id": document_ids, "arg1": args1, "arg2": args2, "label": labels, "t": t, "annotator": annotators}

    df = pd.DataFrame(data)
    return df


def build_convincing_corpus():
    pass


if __name__ == "__main__":
    PATH_DATA_CONVINCING_PROCESSED.mkdir(exist_ok=True, parents=True)
    df = load_convincing()
    # df = pd.read_csv("convincing.csv")

    annotations_per_user = []
    for annotator, group in df.groupby("annotator"):
        annotations_per_user.append((annotator, len(group)))

    k = 3
    for i, (annotator, c) in enumerate(sorted(annotations_per_user, key=lambda x: x[1], reverse=True)):
        if i >= k:
            break

        x = df[df["annotator"] == annotator]
        z_scores = stats.zscore(x["t"])
        abs_z_scores = np.abs(z_scores)

        x = x[abs_z_scores <= 2]

        x.to_csv(PATH_DATA_CONVINCING_PROCESSED / f"convincing_{annotator}_{c}.csv", index=False)
