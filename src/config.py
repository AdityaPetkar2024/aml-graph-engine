from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

# Date-based split. Normal activity stops after Sep 10 (generation artifact),
# so everything from DATA_END onward is dropped.
DATA_END = "2022-09-11"      # exclusive
VAL_START = "2022-09-07"     # train: Sep 1-6
TEST_START = "2022-09-09"    # val: Sep 7-8, test: Sep 9-10
