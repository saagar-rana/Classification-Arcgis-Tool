# -*- coding: utf-8 -*-
"""Split individual-tree LAS datasets into a CSV manifest."""
import os
import random
import math
import csv
import tempfile
import arcpy
import laspy

FIELDS = ["FILE_PATH", "CLASS_NAME", "SPLIT"]


def multivalue(text):
    return list(dict.fromkeys(x.strip().strip("'") for x in (text or "").split(";") if x.strip()))


def discover(roots):
    result, seen = {}, set()
    for root in roots:
        if not os.path.isdir(root):
            raise ValueError(f"Dataset directory does not exist: {root}")
        for folder, _, filenames in os.walk(root):
            for filename in sorted(filenames):
                if not filename.lower().endswith(".las"):
                    continue
                path = os.path.abspath(os.path.join(folder, filename))
                key = os.path.normcase(os.path.realpath(path))
                if key not in seen:
                    seen.add(key)
                    species = os.path.basename(folder)
                    result.setdefault(species, []).append((path, os.path.abspath(root)))
    return result


def validate_las(path, minimum_points=4096):
    if not os.path.isfile(path) or not path.lower().endswith(".las"):
        raise ValueError(f"Not an existing LAS file: {path}")
    with laspy.open(path) as reader:
        count = reader.header.point_count
        if count < minimum_points:
            raise ValueError(f"Fewer than {minimum_points} points: {path}")
        missing = {"X", "Y", "Z", "red", "green", "blue", "intensity"} - set(reader.header.point_format.dimension_names)
        if missing:
            raise ValueError(f"Missing {', '.join(sorted(missing))}: {path}")
    return count


def split_records(by_class, train_fraction, test_fraction, seed):
    if not math.isfinite(train_fraction) or not math.isfinite(test_fraction) or not 0 < train_fraction < 1 or not 0 <= test_fraction < 1 or train_fraction + test_fraction >= 1:
        raise ValueError("Training and testing fractions must leave a positive validation fraction.")
    rng = random.Random(seed)
    rows = []
    for species in sorted(by_class):
        records = sorted(by_class[species])
        rng.shuffle(records)
        minimum = 3 if test_fraction else 2
        if len(records) < minimum:
            raise ValueError(f"{species} needs at least {minimum} suitable files.")
        ntest = max(1, int(len(records) * test_fraction)) if test_fraction else 0
        ntrain = min(max(1, int(len(records) * train_fraction)), len(records) - ntest - 1)
        if ntrain < 1:
            raise ValueError(f"Not enough {species} files for these fractions.")
        for index, (path, dataset, count) in enumerate(records):
            subset = "training" if index < ntrain else ("testing" if ntest and index >= len(records) - ntest else "validation")
            rows.append((path, species, subset, dataset, count, seed, train_fraction, test_fraction))
    if sum(row[2] == "training" for row in rows) < 2:
        raise ValueError("At least two training files are required.")
    return rows
 



class SplitTreeDataset:
    def __init__(self):
        self.label = "Split Tree Dataset"
        self.description = "Create a CSV manifest of training, validation, and optional testing files."
        self.canRunInBackground = False

    def getParameterInfo(self):
        definitions = [
            ("Dataset Directories", "dataset_directories", "DEFolder", "Required", "Input"),
            ("Select Classes", "selected_classes", "GPString", "Required", "Input"),
            ("Training Fraction (0-1)", "training_fraction", "GPDouble", "Required", "Input"),
            ("Create Testing Set", "create_testing_set", "GPBoolean", "Required", "Input"),
            ("Testing Fraction (0-1)", "testing_fraction", "GPDouble", "Optional", "Input"),
            ("Random Seed", "random_seed", "GPLong", "Required", "Input"),
            ("Output Split CSV", "split_csv", "DEFile", "Required", "Output"),
            ("Minimum Points per File", "minimum_points", "GPLong", "Required", "Input"),
        ]
        params = [arcpy.Parameter(displayName=label, name=name, datatype=dtype,
                                  parameterType=required, direction=direction,
                                  multiValue=name in ("dataset_directories", "selected_classes"))
                  for label, name, dtype, required, direction in definitions]
        params[2].value = .8
        params[3].value = False
        params[4].value = .1
        params[4].enabled = False
        params[5].value = 42
        params[6].filter.list = ["csv"]
        params[7].value = 4096
        params[7].filter.type = "Range"
        params[7].filter.list = [1, 2147483647]
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        parameters[4].enabled = bool(parameters[3].value)
        try:
            parameters[1].filter.list = sorted(discover(multivalue(parameters[0].valueAsText)))
        except (OSError, ValueError):
            parameters[1].filter.list = []

    def updateMessages(self, parameters):
        train = parameters[2].value
        test = parameters[4].value if parameters[3].value else 0
        if train is not None and not 0 < float(train) < 1:
            parameters[2].setErrorMessage("Training fraction must be between 0 and 1.")
        if parameters[3].value and (test is None or not 0 < float(test) < 1):
            parameters[4].setErrorMessage("Testing fraction must be between 0 and 1.")
        if train is not None and test is not None and float(train) + float(test) >= 1:
            parameters[2].setErrorMessage("Leave a positive fraction for validation.")
        output = parameters[6].valueAsText
        if output and not output.lower().endswith(".csv"):
            parameters[6].setErrorMessage("Output must use the .csv extension.")
        if parameters[7].value is not None and int(parameters[7].value) < 1:
            parameters[7].setErrorMessage("Minimum points must be at least 1.")

    def execute(self, parameters, messages):
        try:
            self._execute(parameters, messages)
        finally:
            arcpy.ResetProgressor()

    def _execute(self, parameters, messages):
        minimum_points = int(parameters[7].value)
        if minimum_points < 1:
            raise ValueError("Minimum points must be at least 1.")
        arcpy.SetProgressor("default", "Discovering LAS files in selected directories...")
        files = discover(multivalue(parameters[0].valueAsText))
        names = multivalue(parameters[1].valueAsText)
        if not names or any(name not in files for name in names):
            raise ValueError("Select classes present in the dataset directories.")
        valid = {name: [] for name in names}
        total_files = sum(len(files[name]) for name in names)
        processed = 0
        arcpy.SetProgressor("step", "Checking selected LAS files...", 0, total_files, 1)
        for name in names:
            for path, dataset in files[name]:
                arcpy.SetProgressorLabel(f"Checking {processed + 1}/{total_files}: {name} / {os.path.basename(path)}")
                try:
                    valid[name].append((path, dataset, validate_las(path, minimum_points)))
                except Exception as exc:
                    arcpy.AddWarning(str(exc))
                finally:
                    processed += 1
                    arcpy.SetProgressorPosition(processed)
        retained = sum(len(paths) for paths in valid.values())
        arcpy.AddMessage(f"Checked {processed} files: retained {retained}, skipped {processed - retained}. Minimum points: {minimum_points}.")
        arcpy.SetProgressor("default", "Assigning training, validation, and testing splits...")
        train = float(parameters[2].value)
        test = float(parameters[4].value) if parameters[3].value else 0
        if parameters[3].value and not test > 0:
            raise ValueError("Testing fraction must be positive when testing is enabled.")
        rows = split_records(valid, train, test, int(parameters[5].value))
        output = parameters[6].valueAsText
        if not output or not output.lower().endswith(".csv"):
            raise ValueError("Choose an output file with the .csv extension.")
        output = os.path.abspath(output)
        folder = os.path.dirname(output)
        if not os.path.isdir(folder):
            raise ValueError("Select an existing output folder.")
        if os.path.exists(output) and not arcpy.env.overwriteOutput:
            raise ValueError(f"CSV exists; enable overwrite or choose another file: {output}")
        temporary = None
        try:
            arcpy.SetProgressor("step", "Writing split CSV...", 0, len(rows), 1)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                             dir=folder, suffix=".tmp", delete=False) as stream:
                temporary = stream.name
                writer = csv.writer(stream)
                writer.writerow(["FILE_PATH", "CLASS_NAME", "SPLIT", "DATASET_DIR",
                                 "POINT_COUNT", "RANDOM_SEED", "TRAIN_FRAC", "TEST_FRAC", "MIN_POINTS"])
                for index, row in enumerate(rows, start=1):
                    writer.writerow((*row, minimum_points))
                    arcpy.SetProgressorLabel(f"Writing CSV row {index}/{len(rows)}")
                    arcpy.SetProgressorPosition(index)
            os.replace(temporary, output)
        finally:
            if temporary and os.path.exists(temporary):
                os.remove(temporary)
        parameters[6].value = output
        for subset in ("training", "validation", "testing"):
            arcpy.AddMessage(f"{subset}: {sum(row[2] == subset for row in rows)} files")
        arcpy.AddMessage(f"Split CSV: {output}")
        arcpy.AddWarning("Splits are by file; repeated observations of the same tree are not grouped.")

    def postExecute(self, parameters):
        pass
