from pyflowchart import Flowchart,output_html

# Pfad zu deinem Python-Skript
input_datei = "find_variable.py"
output_datei = "flowchart.txt"

# Python-Code einlesen
with open(input_datei, "r") as f:
    code = f.read()

# Flowchart erzeugen
fc = Flowchart.from_code(code)

# Flowchart-Text speichern
# with open(output_datei, "w") as f:
#     f.write(fc.flowchart())

output_html('output.html', 'start', fc.flowchart())
print(f"Flowchart gespeichert")
