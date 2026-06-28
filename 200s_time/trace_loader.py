import sys
sys.path.append("/pub/rricesmi/Arianna/ReflectiveAnalysis")

from HRAStationDataAnalysis.ChiChiHandoff import chi_chi_loader as L

export = L.load_export(
    "/pub/rricesmi/Arianna/ReflectiveAnalysis/HRAStationDataAnalysis/ChiChiHandoff/output/chi_chi_export_3.21.26n3.pkl"
)

records = L.category_records(export, "identified_rcr")

r = records[0]

trace = L.load_trace(export, r)

print(trace.shape)