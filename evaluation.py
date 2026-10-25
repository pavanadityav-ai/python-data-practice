"""Evaluation practice: accuracy, error analysis and duration statistics
on a small sample of speech-recognition style predictions (Telugu words)."""

import csv
import numpy as np

csv_text = """audio_id,duration,predicted,actual
tel_001,4.2,nenu,nenu
tel_002,3.8,vachanu,vellanu
tel_003,5.1,illu,illu
tel_004,2.9,pustakam,pustakam
"""

with open("data.csv", "w") as f:
    f.write(csv_text)

with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    dataset = list(reader)


def report(data):
    predicted = []
    actual = []
    ids = []
    for row in data:
        predicted.append(row["predicted"])
        actual.append(row["actual"])
        ids.append(row["audio_id"])
    predicted = np.array(predicted)
    actual = np.array(actual)
    ids = np.array(ids)
    accuracy = (predicted == actual).mean() * 100
    print(f"Accuracy: {accuracy}%")
    print(f"Wrong: {ids[predicted != actual]}")


def duration_report(data):
    ids = []
    durations = []
    for row in data:
        ids.append(row["audio_id"])
        durations.append(float(row["duration"]))
    ids = np.array(ids)
    durations = np.array(durations)
    print(f"Average: {round(durations.mean(), 2)}")
    print(f"Longest: {ids[durations.argmax()]}")
    print(f"Longer than 4: {ids[durations > 4]} ({len(ids[durations > 4])})")


report(dataset)
duration_report(dataset)
