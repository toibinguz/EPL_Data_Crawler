"""
mapped_epl/mappings.py
Canonical Entity Mappings for EPL (2000/01 - 2026/27) across all 7 raw sources:
1. football-data.co.uk
2. FBref
3. Understat
4. Transfermarkt
5. FotMob (sofascore)
6. FPL API
7. Open-Meteo (stadiums & regions)
"""

import html
import re
import unicodedata
from pathlib import Path
import pandas as pd

# (club_id, name_standard, fd_name, fbref_name, understat_name, tm_club_id, tm_name, fotmob_id, fotmob_names, fpl_short, fpl_names, stadium_file, weather_region)
CLUB_RECORDS = [
    ("ARS", "Arsenal", "Arsenal", "Arsenal", "Arsenal", 11, "Arsenal FC", 9825, ["Arsenal"], "ARS", ["Arsenal"], "Emirates_Stadium.json", "London"),
    ("AVL", "Aston Villa", "Aston Villa", "Aston Villa", "Aston Villa", 405, "Aston Villa", 10252, ["Aston Villa"], "AVL", ["Aston Villa"], "Villa_Park.json", "Birmingham"),
    ("BHA", "Brighton", "Brighton", "Brighton & Hove Albion", "Brighton", 1237, "Brighton & Hove Albion", 10204, ["Brighton & Hove Albion"], "BHA", ["Brighton"], "AMEX_Stadium.json", "Southampton_South"),
    ("BIR", "Birmingham", "Birmingham", None, None, None, "Birmingham City", 8658, ["Birmingham City"], None, [], None, "Birmingham"),
    ("BLB", "Blackburn", "Blackburn", None, None, None, "Blackburn Rovers", 8655, ["Blackburn", "Blackburn Rovers"], None, [], None, "Manchester"),
    ("BLP", "Blackpool", "Blackpool", None, None, None, "Blackpool FC", 8483, ["Blackpool"], None, [], None, "Manchester"),
    ("BOL", "Bolton", "Bolton", None, None, None, "Bolton Wanderers", 8559, ["Bolton Wanderers"], None, [], None, "Manchester"),
    ("BOU", "Bournemouth", "Bournemouth", "Bournemouth", "Bournemouth", 989, "AFC Bournemouth", 8678, ["AFC Bournemouth"], "BOU", ["Bournemouth"], "Vitality_Stadium.json", "Southampton_South"),
    ("BRA", "Bradford", "Bradford", None, None, None, "Bradford City", None, [], None, [], None, "Leeds_Yorkshire"),
    ("BRE", "Brentford", "Brentford", "Brentford", "Brentford", 1148, "Brentford FC", 9937, ["Brentford"], "BRE", ["Brentford"], "Gtech_Community_Stadium.json", "London"),
    ("BUR", "Burnley", "Burnley", "Burnley", "Burnley", 1132, "Burnley FC", 8191, ["Burnley"], "BUR", ["Burnley"], "Turf_Moor.json", "Manchester"),
    ("CAR", "Cardiff", "Cardiff", "Cardiff City", "Cardiff", 603, "Cardiff City", 8344, ["Cardiff City"], None, ["Cardiff"], None, "Wales_Swansea_Cardiff"),
    ("CHA", "Charlton", "Charlton", None, None, None, "Charlton Athletic", None, [], None, [], None, "London"),
    ("CHE", "Chelsea", "Chelsea", "Chelsea", "Chelsea", 631, "Chelsea FC", 8455, ["Chelsea"], "CHE", ["Chelsea"], "Stamford_Bridge.json", "London"),
    ("COV", "Coventry", "Coventry", None, "Coventry", None, None, 8669, ["Coventry City"], None, [], None, "Birmingham"),
    ("CRY", "Crystal Palace", "Crystal Palace", "Crystal Palace", "Crystal Palace", 873, "Crystal Palace", 9826, ["Crystal Palace"], "CRY", ["Crystal Palace"], "Selhurst_Park.json", "London"),
    ("DER", "Derby", "Derby", None, None, None, "Derby County", None, [], None, [], None, "Leicester_Midlands"),
    ("EVE", "Everton", "Everton", "Everton", "Everton", 29, "Everton FC", 8668, ["Everton"], "EVE", ["Everton"], "Goodison_Park.json", "Liverpool"),
    ("FUL", "Fulham", "Fulham", "Fulham", "Fulham", 931, "Fulham FC", 9879, ["Fulham"], "FUL", ["Fulham"], "Craven_Cottage.json", "London"),
    ("HUD", "Huddersfield", "Huddersfield", "Huddersfield Town", "Huddersfield", 1110, "Huddersfield Town", 9796, ["Huddersfield Town"], None, ["Huddersfield"], None, "Leeds_Yorkshire"),
    ("HUL", "Hull", "Hull", None, "Hull", 3008, "Hull City", 8667, ["Hull", "Hull City"], None, ["Hull"], None, "Leeds_Yorkshire"),
    ("IPS", "Ipswich", "Ipswich", "Ipswich Town", "Ipswich", 677, "Ipswich Town", 9902, ["Ipswich Town"], "IPS", ["Ipswich"], "Portman_Road.json", "London"),
    ("LEE", "Leeds", "Leeds", "Leeds United", "Leeds", 399, "Leeds United", 8463, ["Leeds United"], "LEE", ["Leeds"], "Elland_Road.json", "Leeds_Yorkshire"),
    ("LEI", "Leicester", "Leicester", "Leicester City", "Leicester", 1003, "Leicester City", 8197, ["Leicester City"], "LEI", ["Leicester"], "King_Power_Stadium.json", "Leicester_Midlands"),
    ("LIV", "Liverpool", "Liverpool", "Liverpool", "Liverpool", 31, "Liverpool FC", 8650, ["Liverpool"], "LIV", ["Liverpool"], "Anfield.json", "Liverpool"),
    ("LUT", "Luton", "Luton", "Luton Town", "Luton", 1031, "Luton Town", 8346, ["Luton Town"], "LUT", ["Luton"], "Kenilworth_Road.json", "London"),
    ("MCI", "Manchester City", "Man City", "Manchester City", "Manchester City", 281, "Manchester City", 8456, ["Manchester City"], "MCI", ["Man City"], "Etihad_Stadium.json", "Manchester"),
    ("MID", "Middlesbrough", "Middlesbrough", None, "Middlesbrough", 641, "Middlesbrough FC", 8549, ["Middlesbrough"], None, ["Middlesbrough"], None, "Newcastle"),
    ("MUN", "Manchester United", "Man United", "Manchester United", "Manchester United", 985, "Manchester United", 10260, ["Manchester United"], "MUN", ["Man Utd"], "Old_Trafford.json", "Manchester"),
    ("NEW", "Newcastle", "Newcastle", "Newcastle United", "Newcastle United", 762, "Newcastle United", 10261, ["Newcastle", "Newcastle United"], "NEW", ["Newcastle"], "St_James_Park.json", "Newcastle"),
    ("NFO", "Nottingham Forest", "Nott'm Forest", "Nottingham Forest", "Nottingham Forest", 703, "Nottingham Forest", 10203, ["Nottingham Forest"], "NFO", ["Nott'm Forest"], "The_City_Ground.json", "Leicester_Midlands"),
    ("NOR", "Norwich", "Norwich", "Norwich City", "Norwich", 1123, "Norwich City", 9850, ["Norwich", "Norwich City"], "NOR", ["Norwich"], "Carrow_Road.json", "London"),
    ("POR", "Portsmouth", "Portsmouth", None, None, None, "Portsmouth FC", None, [], None, [], None, "Southampton_South"),
    ("QPR", "QPR", "QPR", None, "Queens Park Rangers", 1039, "Queens Park Rangers", 10172, ["Queens Park Rangers"], None, [], None, "London"),
    ("REA", "Reading", "Reading", None, None, 1032, "Reading FC", 9798, ["Reading"], None, [], None, "London"),
    ("SHU", "Sheffield United", "Sheffield United", "Sheffield United", "Sheffield United", 350, "Sheffield United", 8657, ["Sheffield United"], "SHU", ["Sheffield Utd"], "Bramall_Lane.json", "Leeds_Yorkshire"),
    ("SOU", "Southampton", "Southampton", "Southampton", "Southampton", 180, "Southampton FC", 8466, ["Southampton"], "SOU", ["Southampton"], "St_Marys_Stadium.json", "Southampton_South"),
    ("STK", "Stoke", "Stoke", "Stoke City", "Stoke", 512, "Stoke City", 10194, ["Stoke", "Stoke City"], None, ["Stoke"], None, "Leicester_Midlands"),
    ("SUN", "Sunderland", "Sunderland", "Sunderland", "Sunderland", 289, "Sunderland AFC", 8472, ["Sunderland"], "SUN", ["Sunderland"], "Stadium_of_Light.json", "Newcastle"),
    ("SWA", "Swansea", "Swansea", "Swansea City", "Swansea", 2288, "Swansea City", 10003, ["Swansea", "Swansea City"], None, ["Swansea"], None, "Wales_Swansea_Cardiff"),
    ("TOT", "Tottenham", "Tottenham", "Tottenham Hotspur", "Tottenham", 148, "Tottenham Hotspur", 8586, ["Tottenham", "Tottenham Hotspur"], "TOT", ["Spurs", "Tottenham"], "Tottenham_Hotspur_Stadium.json", "London"),
    ("WAT", "Watford", "Watford", "Watford", "Watford", 1010, "Watford FC", 9817, ["Watford"], "WAT", ["Watford"], "Vicarage_Road.json", "London"),
    ("WBA", "West Brom", "West Brom", "West Bromwich Albion", "West Bromwich Albion", 984, "West Bromwich Albion", 8659, ["West Bromwich", "West Bromwich Albion"], None, ["West Brom"], None, "Birmingham"),
    ("WHU", "West Ham", "West Ham", "West Ham United", "West Ham", 379, "West Ham United", 8654, ["West Ham", "West Ham United"], "WHU", ["West Ham"], "London_Stadium.json", "London"),
    ("WIG", "Wigan", "Wigan", None, None, 1071, "Wigan Athletic", 8528, ["Wigan Athletic"], None, [], None, "Manchester"),
    ("WOL", "Wolves", "Wolves", "Wolverhampton Wanderers", "Wolverhampton Wanderers", 543, "Wolverhampton Wanderers", 8602, ["Wolverhampton", "Wolverhampton Wanderers"], "WOL", ["Wolves"], "Molineux_Stadium.json", "Birmingham"),
]


def build_club_lookups():
    fd_to_cid = {}
    fbref_to_cid = {}
    us_to_cid = {}
    tm_id_to_cid = {}
    tm_name_to_cid = {}
    fm_id_to_cid = {}
    fm_name_to_cid = {}
    fpl_to_cid = {}
    stadium_to_cid = {}
    cid_to_region = {}

    for rec in CLUB_RECORDS:
        cid, std, fd, fb, us, tm_id, tm_name, fm_id, fm_names, fpl_s, fpl_names, stad, reg = rec
        cid_to_region[cid] = reg
        if fd:
            fd_to_cid[fd] = cid
        if fb:
            fbref_to_cid[fb] = cid
        if us:
            us_to_cid[us] = cid
        if tm_id is not None:
            tm_id_to_cid[int(tm_id)] = cid
        if tm_name:
            tm_name_to_cid[tm_name] = cid
        if fm_id is not None:
            fm_id_to_cid[int(fm_id)] = cid
        for fn in fm_names:
            fm_name_to_cid[fn] = cid
        if fpl_s:
            fpl_to_cid[fpl_s] = cid
        for fn in fpl_names:
            fpl_to_cid[fn] = cid
        if stad:
            stadium_to_cid[stad] = cid

    # FBref match_shooting Squad short-name aliases
    fbref_to_cid.update({
        "Brighton": "BHA",
        "Wolves": "WOL",
        "Newcastle Utd": "NEW",
        "West Ham": "WHU",
        "Sheffield Utd": "SHU",
        "Manchester Utd": "MUN",
        "Tottenham": "TOT",
        "Nott'ham Forest": "NFO",
        "Huddersfield": "HUD",
        "West Brom": "WBA",
    })

    return {
        "fd": fd_to_cid,
        "fbref": fbref_to_cid,
        "understat": us_to_cid,
        "tm_id": tm_id_to_cid,
        "tm_name": tm_name_to_cid,
        "fm_id": fm_id_to_cid,
        "fm_name": fm_name_to_cid,
        "fpl": fpl_to_cid,
        "stadium": stadium_to_cid,
        "region": cid_to_region,
    }


def export_dimension_csvs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    club_rows = []
    for rec in CLUB_RECORDS:
        cid, std, fd, fb, us, tm_id, tm_name, fm_id, fm_names, fpl_s, fpl_names, stad, reg = rec
        club_rows.append({
            "club_id": cid,
            "name_standard": std,
            "fd_name": fd,
            "fbref_name": fb,
            "understat_name": us,
            "tm_club_id": tm_id,
            "tm_name": tm_name,
            "fotmob_team_id": fm_id,
            "fotmob_names": "|".join(fm_names),
            "fpl_short_name": fpl_s,
            "fpl_names": "|".join(fpl_names),
            "stadium_file": stad,
            "weather_region": reg,
        })
    dim_club_df = pd.DataFrame(club_rows)
    dim_club_df.to_csv(out_dir / "dim_club.csv", index=False)

    season_rows = []
    for y in range(2000, 2027):
        s_str = f"{y}/{str(y + 1)[-2:]}"
        fd_code = f"{str(y)[-2:]}{str(y + 1)[-2:]}"
        season_rows.append({
            "season": s_str,
            "start_year": y,
            "end_year": y + 1,
            "fd_code": fd_code,
        })
    dim_season_df = pd.DataFrame(season_rows)
    dim_season_df.to_csv(out_dir / "dim_season.csv", index=False)
    return dim_club_df, dim_season_df


# Referee normalization (learned from clean/referee_mapping + fixes)
REFEREE_FIXES = {
    "D Gallagh": "D Gallagher",
    "D Gallaghe": "D Gallagher",
    "E Ilderton": "E Ilderton",
    "S Dunn": "S Dunn",
    "A Wiley": "A Wiley",
    "M Messias": "M Messias",
    "U Rennie": "U Rennie",
    "C Foy": "C Foy",
    "R Styles": "R Styles",
    "P Dowd": "P Dowd",
    "H Webb": "H Webb",
    "M Clattenburg": "M Clattenburg",
    "M Dean": "M Dean",
    "M Atkinson": "M Atkinson",
    "A Taylor": "A Taylor",
    "M Oliver": "M Oliver",
    "C Kavanagh": "C Kavanagh",
    "G Scott": "G Scott",
    "J Moss": "J Moss",
    "P Taylor": "P Taylor",
    "R Harris": "R Harris",
    "K Friend": "K Friend",
    "C Pawson": "C Pawson",
    "P Tierney": "P Tierney",
    "S Attwell": "S Attwell",
    "D Coote": "D Coote",
    "J Brooks": "J Brooks",
    "A Madley": "A Madley",
    "R Jones": "R Jones",
    "S Hooper": "S Hooper",
    "T Harrington": "T Harrington",
    "J Gillett": "J Gillett",
    "M Salisbury": "M Salisbury",
    "D England": "D England",
    "T Bramall": "T Bramall",
    "S Barrott": "S Barrott",
    "P Bankes": "P Bankes",
}


def normalize_referee(raw_name: str | None) -> str | None:
    if raw_name is None or not str(raw_name).strip():
        return None
    s = re.sub(r"\s+", " ", str(raw_name).strip())
    # Handle "Dean, M. L" or "Dean, Mike"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            first_init = parts[1][0].upper()
            s = f"{first_init} {parts[0]}"
    else:
        tokens = s.split()
        if len(tokens) >= 2:
            first_init = tokens[0][0].upper()
            last = tokens[-1]
            s = f"{first_init} {last}"
    return REFEREE_FIXES.get(s, s)


SPECIAL_CHAR_MAP = str.maketrans({
    "ł": "l", "Ł": "l",
    "đ": "d", "Đ": "d",
    "ð": "d", "Ð": "d",
    "ø": "o", "Ø": "o",
    "ß": "ss",
    "ı": "i", "İ": "i",
    "æ": "ae", "Æ": "ae",
    "œ": "oe", "Œ": "oe",
})

PLAYER_NAME_ALIASES = {
    "chicharito": "javier hernandez",
    "tino livramento": "valentino livramento",
    "savio": "savinho",
    "zanka": "mathias jorgensen",
    "emi martinez": "emiliano martinez",
    "rob elliot": "robert elliot",
    "zeki fryers": "ezekiel fryers",
    "kepa": "kepa arrizabalaga",
    "jota": "jota peleteiro",
    "nyom": "allan nyom",
    "jose holebas": "jose cholevas",
    "maza": "nasser djiga",
}


def normalize_person_name(name: str | None) -> str:
    """Decode HTML entities, strip accents & special Latin chars, normalize punctuation."""
    if not name:
        return ""
    s = html.unescape(str(name)).translate(SPECIAL_CHAR_MAP)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"['’`\-\.]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return PLAYER_NAME_ALIASES.get(s, s)


def name_tokens(norm_name: str) -> tuple[str, ...]:
    return tuple(norm_name.split()) if norm_name else ()
