import json
import csv
import os
import glob

def load_ketjl():
    path = find_latest_json("ketjl")
    with open(path, "r") as f:
        data = json.load(f)
        results = {}
        for entry in data:
            name = entry["name"]
            results[name] = {
                "mean": entry.get("mean_ns", 0)/ 1e9 if entry.get("mean_ns") is not None else None,
                "mean": entry.get("mean_ns") / 1e9 if entry.get("mean_ns") is not None else None,
                "median": entry.get("median_ns") / 1e9 if entry.get("median_ns") is not None else None,
                "mean": entry.get("mean_ns", 0)/ 1e9 if entry.get("mean_ns") is not None else None,
                "median": entry.get("median_ns") / 1e9 if entry.get("median_ns") is not None else None,
                "ops": entry.get("ops"),
                "source": "ketjl",
            }
        return results


def load_other(source):
    path = find_latest_json(source)
    with open(path, "r") as f:
        data = json.load(f)
        results = {}
        for entry in data["benchmarks"]:
            name = entry["name"]
            stats = entry["stats"]
            results[name] = {
                "mean": stats.get("mean"),
                "median": stats.get("median"),
                "ops": stats.get("ops"),
                "source": source,
            }
        return results


def find_latest_json(library):
    # Option 1: Look for JSON directly under results/<library>/full/
    pattern1 = os.path.join("results", library, "full", "*.json")
    files1 = glob.glob(pattern1)
    if files1:
        latest_file = max(files1, key=os.path.getmtime)
        print(f"Using latest JSON in '{pattern1}': {latest_file}")
        return latest_file

    # Option 2: Look for JSON under any subdirectory
    pattern2 = os.path.join("results", library, "full", "*", "*.json")
    files2 = glob.glob(pattern2)
    if files2:
        latest_file = max(files2, key=os.path.getmtime)
        print(f"Using latest JSON in '{pattern2}': {latest_file}")
        return latest_file

    raise FileNotFoundError(
        f"No benchmark JSON found in '{pattern1}' or '{pattern2}'"
    )

def combine(ketjl, toqito, qutipy):
    keys = set(ketjl) | set(toqito) | set(qutipy)
    table = []

    for name in sorted(keys):
        row = {'name': name}

        row["ketjl_mean_s"] = ketjl.get(name, {}).get("mean")
        row["ketjl_median_s"] = ketjl.get(name, {}).get("median")
        row["ketjl_ops"] = ketjl.get(name, {}).get("ops")

        row["toqito_mean_s"] = toqito.get(name, {}).get("mean")
        row["toqito_median_s"] = toqito.get(name, {}).get("median")
        row["toqito_ops"] = toqito.get(name, {}).get("ops")

        row["qutipy_mean_s"] = qutipy.get(name, {}).get("mean")
        row["qutipy_median_s"] = qutipy.get(name, {}).get("median")
        row["qutipy_ops"] = qutipy.get(name, {}).get("ops")
        table.append(row)

    return table

def main():
    ketjl = load_ketjl()
    toqito = load_other('toqito')
    qutipy = load_other('qutipy')
    
    table = combine(ketjl, toqito, qutipy)

    fields = [
        "name", 
        "ketjl_mean_s", "ketjl_median_s", "ketjl_ops",
        "toqito_mean_s", "toqito_median_s", "toqito_ops",
        "qutipy_mean_s", "qutipy_median_s", "qutipy_ops"
    ]
    with open("combined_results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in table:
            writer.writerow(row)
    print(f"Wrote combined results to combined_results.csv")

if __name__ == '__main__':
    main()