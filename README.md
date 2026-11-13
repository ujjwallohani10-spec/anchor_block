## Penstock Anchor Block Stability (`anchor_block.py`)

Preliminary stability check of a penstock anchor block, based on IS 11639 (Part 2) and IS 1893 (seismic).

**Checks** (XY and ZY planes, static and seismic, pipe expansion and contraction):
overturning, sliding, eccentricity (middle-third rule) and bearing pressure.

**Method**
- Forces: pipe and water weight, hydrostatic force at the bend, dynamic force from the change of flow direction, earth pressure on the upstream face, block weight, overburden, and friction at the supports or expansion joint
- Earth pressure: Coulomb (static) and Mononobe-Okabe (seismic)
- Factors of safety are configurable (`SafetyFactors`)

**Run**
```
pip install pandas
python anchor_block.py
```

To use your own data, edit the dataclass defaults, or build the calculator from a dictionary with `build_from_dict(params)`.

**Assumptions and limitations**
- Friction at the supports or joint acts downstream in expansion and upstream in contraction
- The ZY overturning check uses the same earth-pressure moment as XY (conservative simplification)
- Example inputs are illustrative, not from a real project
- For learning and preliminary checks only, not for construction use
