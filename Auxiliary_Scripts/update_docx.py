from docx import Document
import os

with open('/Users/rahul/Documents/HybridCrowdFramework/Thesis_Documentation/generate_docx.py', 'r') as f:
    code = f.read()

table_code = """
add_image(doc, os.path.join(img_dir, "Final_Dash_3.png"))
add_caption(doc, "Figure 5.13: Automatic Transition to CSRNet Jet Colormap Density Heatmap for Dense Crowds Exceeding Occupancy Threshold")

add_paragraph(doc, "To quantitatively substantiate the continuous tracking capability, Table 5.6 presents an exact extraction from the local telemetry logs (anomaly_alerts.csv), demonstrating the system's ability to persistently calculate absolute counts, relative density percentage, and dynamic motion scores during dense crowd congestion.")
add_caption(doc, "Table 5.6: Extracted Telemetry Log Sample During Dense Crowd Processing State")

table2 = doc.add_table(rows=5, cols=5)
table2.style = 'Table Grid'
hdr_cells = table2.rows[0].cells
hdr_cells[0].text = 'Timestamp'
hdr_cells[1].text = 'Risk Level'
hdr_cells[2].text = 'Recorded Count'
hdr_cells[3].text = 'Density (%)'
hdr_cells[4].text = 'Motion Score'

row_data = [
    ("2026-03-22 23:34:18", "MODERATE", "309", "77.37%", "4.41"),
    ("2026-03-22 23:34:26", "MODERATE", "312", "78.24%", "5.19"),
    ("2026-03-22 23:51:14", "MODERATE", "331", "82.77%", "3.25"),
    ("2026-03-22 23:36:21", "HIGH RISK", "450", "99.90%", "15.50")
]

for i, data in enumerate(row_data):
    row_cells = table2.rows[i+1].cells
    row_cells[0].text = data[0]
    row_cells[1].text = data[1]
    row_cells[2].text = data[2]
    row_cells[3].text = data[3]
    row_cells[4].text = data[4]
"""

old_code = """add_image(doc, os.path.join(img_dir, "Final_Dash_3.png"))
add_caption(doc, "Figure 5.13: Automatic Transition to CSRNet Jet Colormap Density Heatmap for Dense Crowds Exceeding Occupancy Threshold")"""

code = code.replace(old_code, table_code)

with open('/Users/rahul/Documents/HybridCrowdFramework/Thesis_Documentation/generate_docx.py', 'w') as f:
    f.write(code)
