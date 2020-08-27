PIPELINE_DATA_FILE = 'pipeline_data'

# raw data saved here:
# 0 = filename, 1 = year from, 2 = year to
RAW_DATA_FILE_PATH = 'data/01_raw/{0}_from_{1}_to_{2}.csv'
CLEAN_DATA_FILE_PATH = 'data/02_clean/{0}_from_{1}_to_{2}.csv'

MODEL_OUTPUT_FOLDER = 'data/03_output/'

# urls
WTA_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{}.csv"
ITF_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_qual_itf_{}.csv"

# Carpet to be treated as grass
SURFACE_MAP = {
    'Hard': 'Hard',
    'Clay': 'Clay',
    'Grass': 'Grass',
    'Carpet': 'Grass'
}

ROUND_ORDER = [
    'Q1', 'Q2', 'Q3',  # qualifiers
    'RR',  # round robin
    'R128', 'R64', 'R32',
    'R16', 'QF', 'SF',
    'BR',  # bronze round
    'F'
]

# various columns, jeff sackman columns prefixed with J
SOURCE_COL = 'source'

J_WINNER_COL = 'winner_name'
J_LOSER_COL = 'loser_name'
J_SCORE_COL = 'score'
J_SURFACE_COL = 'surface'
J_T_DATE = 'tourney_date'
J_T_NAME = 'tourney_name'
J_ROUND = 'round'

PARAMS = {
    'K': 269.95,
    'offset': 1,
    'shape': .261,
    'surface_weight': .121,
    'itf_deduction': .339,
    'p': .53,
    'straight_sets_boost': .072,
    'trend_rate': .285,
    'trend_weight': .0736,
    'all_time_weight': .598,
}
