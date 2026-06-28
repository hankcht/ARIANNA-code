import sys
import numpy as np
sys.path.append("/pub/rricesmi/Arianna/ReflectiveAnalysis")
import tensorflow
print(tensorflow.__version__)
from tensorflow import keras


from HRAStationDataAnalysis.ChiChiHandoff import chi_chi_loader as L

export = L.load_export(
    "/pub/rricesmi/Arianna/ReflectiveAnalysis/HRAStationDataAnalysis/ChiChiHandoff/output/chi_chi_export_3.21.26n3.pkl"
)


# 1. get pass BL events
records = L.category_records(export, "pass_bl")

r = records[0]

# 2. load all traces
traces = L.load_traces(
    export,
    records,
    nurfiles_folder="/pub/rricesmi/Arianna/ReflectiveAnalysis/HRAStationDataAnalysis/StationData/nurFiles/9.1.25"
)

# 3. convert to array
traces = np.array(traces, dtype=object)

print(len(traces))
print(traces[0].shape)

# overwrite for specific run
timestamp =  '12.16.25_14-53' # 11.26.25_13-52
model_path = f'/dfs8/sbarwick_lab/ariannaproject/tangch3/HGQ2/{timestamp}/models/'
# print(f"Loading model: {model_path}")
# model = keras.models.load_model(f'{model_path}12.16.25_14-53_HGQ2_model.keras')
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.utils import custom_object_scope
from hgq.layers import QConv2D, QDense  # wherever your Q-layers are defined

custom_objects = {
    'InputLayer': InputLayer,
    'QConv2D': QConv2D,
    'QDense': QDense,
    # add gradient_reversal_operation if DANN
}

with custom_object_scope(custom_objects):
    model = keras.models.load_model(f'{model_path}{timestamp}_HGQ2_model.keras', compile=False)

prefix = 'hgq'

traces = traces.squeeze(-1).transpose(0, 2, 1)
prob = model.predict(traces).flatten()

from refactor_checks import plot_histogram
amp='both'
plot_histogram(prob, amp=amp, timestamp=timestamp, prefix="passing_BL")