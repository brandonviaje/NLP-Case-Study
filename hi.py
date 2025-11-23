import json

path = "Sentiment_Analysis.ipynb"  

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove `metadata.widgets` if it exists
if "widgets" in nb.get("metadata", {}):
    del nb["metadata"]["widgets"]
    print("Removed broken widget metadata.")
else:
    print("No widget metadata found.")

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook fixed.")