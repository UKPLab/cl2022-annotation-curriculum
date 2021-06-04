import datetime
import json
import os
import statistics
import zipfile
from collections import defaultdict
from typing import Dict, List

import attr

EventList = List[Dict]
ROOT = os.path.dirname(__file__)
PATH = os.path.join(ROOT, "tsv")


def load_events(path_to_project_zip: str) -> EventList:
    events = []

    with zipfile.ZipFile(os.path.join(PATH, path_to_project_zip), "r") as myzip:
        with myzip.open("event.log") as f:
            for line in f:
                event = json.loads(line)
                events.append(event)

    return events


def group_by(field: str, events: EventList) -> Dict[str, EventList]:
    grouped_events = defaultdict(list)

    for event in events:
        value = event[field]
        grouped_events[value].append(event)

    return grouped_events


def analyze_single_project(project_name: str, path_to_project_zip: str) -> Dict:
    events = load_events(path_to_project_zip)
    events_by_user = group_by("user", events)

    results = {}
    users = ["valentina.jung", "nederstigt", "pablo.schaeffner", "regina.schoenberger"]

    for user, user_events in events_by_user.items():

        if user not in users:
            continue

        annotation_time_total = calculate_annotation_time_total(user_events, user)

        annotation_time_span = calculate_annotation_time_by_span_created(user_events, user)

        annotation_time_feature_value = calculate_annotation_time_by_feature_value_updated(user_events, user)

        results[user] = {
            "annotation_time_span": annotation_time_span,
            "annotation_time_feature_value": annotation_time_feature_value,
            "annotation_time_total": annotation_time_total,
        }

    return results


def calculate_annotation_time_total(events: EventList, username: str) -> Dict:
    differences = {}
    docs = {}

    for event in events:
        if event["user"] != username:
            continue
        try:
            event["document_name"]
        except KeyError:
            continue
        try:
            docs[event["document_name"]].append(int(event["created"] / 1000))
        except KeyError:
            docs[event["document_name"]] = [int(event["created"] / 1000)]

    for k, v in docs.items():
        differences[k] = get_max_timedelta(v)
    return differences


def get_max_timedelta(times: List) -> datetime.datetime:
    min_time = datetime.datetime.fromtimestamp((min(times)))
    max_time = datetime.datetime.fromtimestamp((max(times)))
    delta = max_time - min_time
    return delta.total_seconds()  # , max_time, min_time


def calculate_annotation_time_by_span_created(events: EventList, username: str) -> Dict:
    span_created_events = [e for e in events if e["event"] == "SpanCreatedEvent" and e["annotator"] == username]
    differences = {}

    for group in group_by("document_name", span_created_events).values():
        differences[group[0]["document_name"]] = []
        for i in range(len(group) - 1):
            e1, e2 = group[i], group[i + 1]
            time_difference = (e2["created"] - e1["created"]) / 1000.0

            if time_difference > 90:
                continue

            differences[group[0]["document_name"]].append(time_difference)

    return differences


def calculate_annotation_time_by_feature_value_updated(events: EventList, username: str) -> Dict:
    span_created_events = {}
    feature_value_updated_events = {}

    target_events = [e for e in events if e["event"] == "SpanCreatedEvent" or e["event"] == "FeatureValueUpdatedEvent"]

    last_event_was_span_created = False

    for group in group_by("document_name", target_events).values():
        for e in group:
            details = e.get("details", {})
            if e["event"] == "SpanCreatedEvent":
                key = (details["begin"], details["end"])
                span_created_events[key] = e
                last_event_was_span_created = True
            elif e["event"] == "FeatureValueUpdatedEvent" and last_event_was_span_created:
                annotation = details["annotation"]
                key = (annotation["begin"], annotation["end"])

                if key not in feature_value_updated_events:
                    feature_value_updated_events[key] = e

                last_event_was_span_created = False
            else:
                last_event_was_span_created = False

    times = {}

    for span, span_created_event in span_created_events.items():
        if span not in feature_value_updated_events:
            continue

        if username != span_created_event["annotator"]:
            continue

        feature_value_updated_event = feature_value_updated_events[span]
        time_difference = (feature_value_updated_event["created"] - span_created_event["created"]) / 1000.0

        if time_difference < 2 or time_difference > 55:
            continue

        assert time_difference > 0, (span_created_event, feature_value_updated_event)

        try:
            times[span_created_event["document_name"]].append(time_difference)
        except KeyError:
            times[span_created_event["document_name"]] = [time_difference]

    return times


def write_data_annotation_time_span(data: dict):
    with open(list(data.keys())[0] + "_gleipnir_annotation_time_span.tsv", "w") as outlog:
        outlog.write("File\tTime Total\tTime Span\tFeature Value\n")
        username = list(data.keys())[0]
        for key in data[username]["annotation_time_span"].keys():
            try:
                outlog.write(
                    "{}\t{}\t{}\t{}\n".format(
                        key,
                        data[username]["annotation_time_total"][key],
                        sum(data[username]["annotation_time_span"][key]),
                        sum(data[username]["annotation_time_feature_value"][key]),
                    )
                )
            except KeyError:
                # timespan is always longer than feature value:
                outlog.write(
                    "{}\t{}\t{}\t{}\n".format(
                        key,
                        data[username]["annotation_time_total"][key],
                        sum(data[username]["annotation_time_span"][key]),
                        0,
                    )
                )


def main():
    results_project1 = analyze_single_project(
        "project1", "ClS%3A+Medizin+-+EDAs+%283rd+-+jung%29_project_2020-08-27_1157.zip"
    )
    results_project2 = analyze_single_project(
        "project2", "ClS%3A+Medizin+-+EDAs+%283rd+-+nederstigt%29_project_2020-08-27_1157.zip"
    )
    results_project3 = analyze_single_project(
        "project3", "ClS%3A+Medizin+-+EDAs+%283rd+-+schaeffner%29_project_2020-08-27_1157.zip"
    )
    results_project4 = analyze_single_project(
        "project4", "ClS%3A+Medizin+-+EDAs+%283rd+-+schoenberger%29_project_2020-08-27_1157.zip"
    )

    write_data_annotation_time_span(results_project1)
    write_data_annotation_time_span(results_project2)
    write_data_annotation_time_span(results_project3)
    write_data_annotation_time_span(results_project4)


if __name__ == "__main__":
    main()
