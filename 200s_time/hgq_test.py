import hgq
from hgq.utils import sugar

print("HGQ version:", hgq.__version__)

from hgq.utils.cost import *
print("Available cost attributes:", dir())