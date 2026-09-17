import os
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from fredapi import Fred
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
FRED_KEY = os.getenv("FRED_API_KEY")

CONCEPT_KEYWORDS = {

    "jet fuel": {
        "keywords": ["airline", "aircraft", "aviation", "flight", "carrier", "passengers"],
        "direction": "negative",
        "lag_days": 21
    },
    "oil price": {
        "keywords": ["oil", "petroleum", "crude", "drilling", "upstream", "downstream", "refinery", "exploration"],
        "direction": "positive",
        "lag_days": 5
    },
    "natural gas": {
        "keywords": ["natural gas", "lng", "pipeline", "gas production"],
        "direction": "positive",
        "lag_days": 5
    },

    "semiconductor": {
        "keywords": ["semiconductor", "chip", "foundry", "wafer", "fabless", "integrated circuit"],
        "direction": "positive",
        "lag_days": 30
    },
    "shipping": {
        "keywords": ["logistics", "freight", "shipping", "supply chain", "warehouse", "distribution", "import", "export"],
        "direction": "negative",
        "lag_days": 21
    },
    "manufacturing output": {
        "keywords": ["manufactur", "factory", "production", "assembly", "industrial"],
        "direction": "positive",
        "lag_days": 30
    },
    "lithium": {
        "keywords": ["battery", "electric vehicle", "ev", "lithium"],
        "direction": "negative",
        "lag_days": 30
    },
    "steel": {
        "keywords": ["steel", "metal fabrication", "construction materials"],
        "direction": "negative",
        "lag_days": 30
    },
    "copper": {
        "keywords": ["copper", "wiring", "electrical components"],
        "direction": "negative",
        "lag_days": 30
    },

    "consumer confidence": {
        "keywords": ["retail", "consumer", "e-commerce", "shopping", "store", "merchandise"],
        "direction": "positive",
        "lag_days": 21
    },
    "personal savings rate": {
        "keywords": ["discretionary", "luxury", "apparel", "leisure"],
        "direction": "negative",
        "lag_days": 30
    },
    "ecommerce": {
        "keywords": ["online retail", "e-commerce", "marketplace", "digital sales"],
        "direction": "positive",
        "lag_days": 14
    },
    "restaurant sales": {
        "keywords": ["restaurant", "dining", "food service", "quick service"],
        "direction": "positive",
        "lag_days": 14
    },

    "interest rate": {
        "keywords": ["bank", "lending", "loan", "mortgage", "credit", "deposit", "interest rate"],
        "direction": "negative",
        "lag_days": 14
    },
    "yield curve": {
        "keywords": ["bank", "financial institution", "insurance", "investment bank"],
        "direction": "positive",
        "lag_days": 60
    },
    "credit spread": {
        "keywords": ["bond", "fixed income", "credit rating", "debt issuance"],
        "direction": "negative",
        "lag_days": 30
    },
    "trading volume": {
        "keywords": ["brokerage", "trading", "exchange", "asset management"],
        "direction": "positive",
        "lag_days": 7
    },

    "fda approval": {
        "keywords": ["pharmaceutical", "biotech", "drug", "clinical trial", "fda", "therapeutic"],
        "direction": "positive",
        "lag_days": 30
    },
    "healthcare spending": {
        "keywords": ["healthcare", "hospital", "medical device", "patient care", "insurance"],
        "direction": "positive",
        "lag_days": 60
    },
    "drug prices": {
        "keywords": ["generic drug", "pharmacy", "prescription"],
        "direction": "negative",
        "lag_days": 30
    },

    "mortgage rate": {
        "keywords": ["real estate", "reit", "property", "housing", "construction"],
        "direction": "negative",
        "lag_days": 30
    },
    "housing starts": {
        "keywords": ["homebuilder", "residential construction", "home sales"],
        "direction": "positive",
        "lag_days": 60
    },
    "commercial real estate": {
        "keywords": ["office space", "commercial property", "retail space", "industrial property"],
        "direction": "negative",
        "lag_days": 60
    },

    "air travel demand": {
        "keywords": ["airline", "airport", "travel", "tourism", "vacation"],
        "direction": "positive",
        "lag_days": 14
    },
    "hotel occupancy": {
        "keywords": ["hotel", "resort", "hospitality", "lodging"],
        "direction": "positive",
        "lag_days": 21
    },
    "gambling": {
        "keywords": ["casino", "gaming", "betting", "wager"],
        "direction": "positive",
        "lag_days": 14
    },

    "tech spending": {
        "keywords": ["software", "cloud", "saas", "enterprise", "data centre", "platform"],
        "direction": "positive",
        "lag_days": 30
    },
    "vix": {
        "keywords": ["technology", "growth stock", "internet"],
        "direction": "negative",
        "lag_days": 5
    },

    "power consumption": {
        "keywords": ["utility", "electric", "power generation", "grid"],
        "direction": "positive",
        "lag_days": 30
    },
    "renewable energy": {
        "keywords": ["solar", "wind", "renewable", "clean energy"],
        "direction": "positive",
        "lag_days": 30
    },

    "dollar index": {
        "keywords": ["international", "global", "multinational", "foreign", "overseas", "export"],
        "direction": "negative",
        "lag_days": 30
    },
    "china": {
        "keywords": ["china", "asia pacific", "manufacturing partner"],
        "direction": "positive",
        "lag_days": 30
    },
    "emerging market": {
        "keywords": ["emerging market", "developing economies", "global expansion"],
        "direction": "positive",
        "lag_days": 30
    },

    "consumer sentiment": {
        "keywords": ["brand", "market share", "competitive", "consumer goods"],
        "direction": "positive",
        "lag_days": 21
    },
}

def get_driving_factors(ticker, company_info):

    print("\n  [Rule-based] Analysing company profile for driving factors...")

    description = (company_info.get("longBusinessSummary", "") or "").lower()
    industry = (company_info.get("industry", "") or "").lower()
    sector = (company_info.get("sector", "") or "").lower()

    full_text = f"{description} {description} {industry} {sector}"

    matched_factors = []

    for concept, data in CONCEPT_KEYWORDS.items():
        match_count = 0
        matched_keywords = []

        for keyword in data["keywords"]:
            if keyword in full_text:
                match_count += 1
                matched_keywords.append(keyword)

        if match_count > 0:
            matched_factors.append({
                "factor": concept,
                "direction": data["direction"],
                "rationale": f"Matched keywords: {', '.join(matched_keywords)}",
                "data_concept": concept,
                "estimated_lag_days": data["lag_days"],
                "priority": "high" if match_count >= 2 else "medium",
                "match_score": match_count
            })

    matched_factors = sorted(
        matched_factors, key=lambda x: x["match_score"], reverse=True
    )[:8]

    if not matched_factors:
        print("  [Rule-based] No keyword matches found — using sector fallback")

        matched_factors = [
            {
                "factor": "consumer confidence",
                "direction": "positive",
                "rationale": "Default macro signal — no specific keywords matched",
                "data_concept": "consumer confidence",
                "estimated_lag_days": 21,
                "priority": "medium",
                "match_score": 0
            },
            {
                "factor": "interest rate",
                "direction": "negative",
                "rationale": "Default macro signal — no specific keywords matched",
                "data_concept": "interest rate",
                "estimated_lag_days": 14,
                "priority": "medium",
                "match_score": 0
            }
        ]

    print(f"  [Rule-based] Identified {len(matched_factors)} driving factors:")
    for f in matched_factors:
        direction = "↑" if f["direction"] == "positive" else "↓"
        priority_marker = "★" if f["priority"] == "high" else "·"
        print(f"  {priority_marker} {direction} {f['factor']:<35} "
              f"(~{f['estimated_lag_days']}d lag, {f['match_score']} matches)")

    return {"company": ticker, "factors": matched_factors}

FRED_CONCEPT_MAP = {

    "oil": "DCOILWTICO",
    "crude oil": "DCOILWTICO",
    "oil price": "DCOILWTICO",
    "natural gas": "DHHNGSP",
    "gasoline": "GASREGCOVW",
    "jet fuel": "DJFUELUSGULF",
    "energy": "DCOILWTICO",
    "fuel": "DJFUELUSGULF",
    "heating oil": "DHOILNYH",
    "propane": "DPROPANEMBTX",
    "coal": "COALNGSTMMUS",
    "electricity": "APU000072610",
    "energy prices": "PPIENG",
    "energy production": "IPG2211A2N",
    "renewable energy": "ELETTNETUS",

    "interest rate": "FEDFUNDS",
    "federal funds": "FEDFUNDS",
    "fed rate": "FEDFUNDS",
    "yield curve": "T10Y2Y",
    "2yr 10yr spread": "T10Y2Y",
    "yield curve inversion": "T10Y2Y",
    "treasury": "DGS10",
    "10 year yield": "DGS10",
    "10yr treasury": "DGS10",
    "2 year yield": "DGS2",
    "2yr treasury": "DGS2",
    "30 year yield": "DGS30",
    "real interest rate": "DFII10",
    "tips yield": "DFII10",
    "credit spread": "BAMLH0A0HYM2",
    "high yield spread": "BAMLH0A0HYM2",
    "junk bond spread": "BAMLH0A0HYM2",
    "investment grade spread": "BAMLC0A0CM",
    "corporate bond spread": "BAMLC0A0CM",
    "mortgage rate": "MORTGAGE30US",
    "30yr mortgage": "MORTGAGE30US",
    "15yr mortgage": "MORTGAGE15US",
    "libor": "USDONTD156N",
    "sofr": "SOFR",
    "prime rate": "DPRIME",
    "ted spread": "TEDRATE",
    "bank lending rate": "MPRIME",
    "commercial paper": "DCPF3M",

    "inflation": "CPIAUCSL",
    "cpi": "CPIAUCSL",
    "core inflation": "CPILFESL",
    "core cpi": "CPILFESL",
    "pce": "PCEPI",
    "core pce": "PCEPILFE",
    "producer price": "PPIACO",
    "ppi": "PPIACO",
    "import prices": "IR",
    "export prices": "IQ",
    "food prices": "CPIUFDSL",
    "housing inflation": "CPIHOSSL",
    "medical inflation": "CPIMEDSL",
    "wage inflation": "CES0500000003",
    "wage growth": "CES0500000003",
    "average hourly earnings": "CES0500000003",

    "unemployment": "UNRATE",
    "jobless claims": "ICSA",
    "initial claims": "ICSA",
    "continuing claims": "CCSA",
    "job openings": "JTSJOL",
    "jolts": "JTSJOL",
    "quit rate": "JTSQUR",
    "layoffs": "JTSLDL",
    "hiring rate": "JTSHIL",
    "nonfarm payroll": "PAYEMS",
    "jobs": "PAYEMS",
    "employment": "PAYEMS",
    "labour force participation": "CIVPART",
    "underemployment": "U6RATE",
    "long term unemployment": "UEMPLT27",
    "manufacturing jobs": "MANEMP",
    "construction jobs": "USCONS",
    "retail jobs": "USTRADE",
    "tech jobs": "USINFO",
    "healthcare jobs": "USHLTH",
    "financial jobs": "USFIRE",
    "government jobs": "USGOVT",

    "consumer confidence": "UMCSENT",
    "consumer sentiment": "UMCSENT",
    "consumer expectations": "MICH",
    "retail sales": "RSXFS",
    "retail sales ex auto": "RSFSXMV",
    "personal spending": "PCE",
    "personal consumption": "PCE",
    "personal income": "PI",
    "disposable income": "DSPIC96",
    "personal savings rate": "PSAVERT",
    "personal savings": "PSAVERT",
    "credit card debt": "REVOLSL",
    "consumer credit": "TOTALSL",
    "auto loans": "DTCTHFNM",
    "student loans": "SLOAS",
    "household debt": "HDTGPDUSQ163N",
    "credit card delinquency": "DRCCLACBS",
    "auto loan delinquency": "DRAUTOACBS",
    "mortgage delinquency": "DRSFRMACBS",
    "consumer delinquency": "DRCCLACBS",
    "bankruptcy": "BAPCPCHG",

    "bank lending": "TOTLL",
    "loan growth": "TOTLL",
    "commercial loans": "BUSLOANS",
    "commercial real estate loans": "CREACBW027SBOG",
    "residential loans": "RREACBW027SBOG",
    "small business loans": "SBODP",
    "bank credit": "TOTBKCR",
    "money supply m2": "M2SL",
    "money supply m1": "M1SL",
    "monetary base": "BOGMBASE",
    "bank reserves": "RESBALNS",
    "excess reserves": "EXCSRESNS",
    "financial stress": "STLFSI",
    "financial conditions": "NFCI",
    "chicago fed": "NFCI",
    "kansas city fed": "KCFSI",
    "credit tightening": "DRTSCILM",
    "lending standards": "DRTSCILM",
    "bank profitability": "USNIM",
    "net interest margin": "USNIM",

    "gdp": "GDP",
    "gdp growth": "A191RL1Q225SBEA",
    "real gdp": "GDPC1",
    "nominal gdp": "GDP",
    "industrial production": "INDPRO",
    "capacity utilisation": "TCU",
    "manufacturing pmi": "MANEMP",
    "ism manufacturing": "NAPM",
    "ism services": "NMFCI",
    "business investment": "PNFIA",
    "government spending": "FGEXPND",
    "trade balance": "BOPGSTB",
    "current account": "BOPCA",
    "exports": "EXPGS",
    "imports": "IMPGS",

    "housing starts": "HOUST",
    "building permits": "PERMIT",
    "home sales": "HSN1F",
    "existing home sales": "EXHOSLUSM495S",
    "pending home sales": "MSACSR",
    "case shiller": "CSUSHPISA",
    "home prices": "CSUSHPISA",
    "house prices": "CSUSHPISA",
    "rent": "CUUR0000SEHA",
    "rental prices": "CUUR0000SEHA",
    "homeownership rate": "RHORUSQ156N",
    "vacancy rate": "RRVRUSQ156N",
    "commercial real estate": "COMREACBW027SBOG",
    "office vacancy": "COMREACBW027SBOG",
    "mortgage applications": "MORTGAGE30US",
    "housing affordability": "FIXHAI",
    "construction spending": "TTLCONS",
    "lumber prices": "WPU0811",

    "dollar": "DTWEXBGS",
    "usd": "DTWEXBGS",
    "dollar index": "DTWEXBGS",
    "dollar strength": "DTWEXBGS",
    "euro": "DEXUSEU",
    "yen": "DEXJPUS",
    "yuan": "DEXCHUS",
    "pound": "DEXUSUK",
    "canadian dollar": "DEXCAUS",
    "emerging market currency": "DTWEXEMEGS",
    "china pmi": "CHNPMICNEFON",
    "eurozone pmi": "EURPMICNEFON",
    "japan pmi": "JPNPMICNEFON",
    "global pmi": "GHPMINMANINDMEI",
    "china industrial production": "CHNIPMEI",
    "germany industrial production": "DEUITPIOFMEI",
    "uk gdp": "GBKPICNBP16",
    "eurozone gdp": "CPMNACSCAB1GQEU272S",
    "japan gdp": "JPNRGDPEXP",
    "global trade": "XTEXVA01USM667S",
    "world trade": "XTEXVA01USM667S",

    "copper": "PCOPPUSDM",
    "gold": "GOLDPMGBD228NLBM",
    "silver": "SLVPRUSD",
    "platinum": "PLATINUMUSDM",
    "palladium": "PALUSDM",
    "steel": "WPUSI019011",
    "iron ore": "PIORECRUSDM",
    "aluminum": "PALUMUSDM",
    "nickel": "PNICKUSDM",
    "zinc": "PZINCUSDM",
    "lead": "PLEADUSDM",
    "tin": "PTINUSDM",
    "lithium": "LITHIUMUSDM",
    "cobalt": "PCOBAUSDM",
    "rare earth": "WPUSI019011",
    "corn": "PMAIZMTUSDM",
    "wheat": "PWHEAMTUSDM",
    "soybean": "PSOYBUSDM",
    "cotton": "PCOTTINDUSDM",
    "coffee": "PCOFFOTMUSDM",
    "sugar": "PSUGAISAUSDM",
    "cocoa": "PCOCOUSDM",
    "rubber": "PRUBBINDM",
    "shipping": "BDIY",
    "baltic dry": "BDIY",
    "freight": "BDIY",
    "container shipping": "BDIY",
    "supply chain": "BDIY",

    "semiconductor": "IPG3344S",
    "chip": "IPG3344S",
    "chip production": "IPG3344S",
    "tech production": "IPG3344S",
    "tech spending": "Y033RC1Q027SBEA",
    "it spending": "Y033RC1Q027SBEA",
    "software spending": "Y033RC1Q027SBEA",
    "data centre": "IPG3344S",
    "cloud computing": "Y033RC1Q027SBEA",
    "semiconductor inventory": "AMINVINDNS",
    "tech inventory": "AMINVINDNS",
    "pc sales": "IPG3344S",
    "smartphone": "IPG3344S",

    "airline passengers": "AIR",
    "air travel": "AIR",
    "air freight": "RAILFRTCARLOADSD11",
    "travel demand": "AIR",
    "hotel occupancy": "TRVLTRNS",
    "tourism": "TRVLTRNS",
    "international travel": "TRVLTRNS",

    "bank credit": "TOTBKCR",
    "bank loans": "TOTLL",
    "deposit growth": "DPSACBW027SBOG",
    "bank deposits": "DPSACBW027SBOG",
    "net interest margin": "USNIM",
    "bank profitability": "USNIM",
    "trading volume": "NYFEDTRADE",
    "ipo activity": "NYFEDTRADE",
    "m&a activity": "NYFEDTRADE",
    "private equity": "NYFEDTRADE",
    "venture capital": "NYFEDTRADE",

    "healthcare spending": "HLTHSCPCHCSA",
    "medical costs": "HLTHSCPCHCSA",
    "drug prices": "WPUSI07311",
    "pharmaceutical prices": "WPUSI07311",
    "hospital admissions": "HLTHSCPCHCSA",
    "insurance costs": "HLTHSCPCHCSA",
    "medicare spending": "HLTHSCPCHCSA",
    "medicaid spending": "HLTHSCPCHCSA",
    "r&d spending": "Y033RC1Q027SBEA",
    "biotech funding": "Y033RC1Q027SBEA",

    "food prices": "CPIUFDSL",
    "grocery prices": "CPIUFDSL",
    "restaurant sales": "MRTSSM722USS",
    "restaurant spending": "MRTSSM722USS",
    "ecommerce": "ECOMSA",
    "online retail": "ECOMSA",
    "auto sales": "TOTALSA",
    "vehicle sales": "TOTALSA",
    "truck sales": "LAUTOSA",

    "manufacturing output": "IPMAN",
    "factory orders": "AMTMNO",
    "durable goods": "DGORDER",
    "capital goods": "NEWORDER",
    "inventory": "ISRATIO",
    "inventory sales ratio": "ISRATIO",
    "supply chain disruption": "ISRATIO",
    "supplier deliveries": "NAPMSD",
    "backlog orders": "AMOBNO",
    "unfilled orders": "AMOBNO",

    "reit": "WILL5000IND",
    "property prices": "CSUSHPISA",
    "commercial property": "COMREACBW027SBOG",
    "office space": "COMREACBW027SBOG",
    "industrial property": "COMREACBW027SBOG",
    "retail property": "COMREACBW027SBOG",

    "power consumption": "IPG2211A2N",
    "electricity demand": "IPG2211A2N",
    "utility output": "IPG2211A2N",
    "nuclear energy": "IPG2211A2N",
    "solar capacity": "ELETTNETUS",
    "wind capacity": "ELETTNETUS",
    "renewable capacity": "ELETTNETUS",
    "carbon price": "DCOILWTICO",

    "vix": "VIXCLS",
    "volatility": "VIXCLS",
    "fear index": "VIXCLS",
    "market volatility": "VIXCLS",
    "investor sentiment": "UMCSENT",
    "business confidence": "BSCICP03USM665S",
    "business sentiment": "BSCICP03USM665S",
    "ceo confidence": "BSCICP03USM665S",
    "small business confidence": "NFIB",
    "nfib": "NFIBSI",
    "geopolitical risk": "VIXCLS",
    "political uncertainty": "VIXCLS",
    "policy uncertainty": "USEPUINDXD",
    "economic policy uncertainty": "USEPUINDXD",
    "uncertainty": "USEPUINDXD",
}

YFINANCE_CONCEPT_MAP = {

    "oil price": "CL=F",
    "crude oil": "CL=F",
    "wti crude": "CL=F",
    "brent crude": "BZ=F",
    "natural gas": "NG=F",
    "gasoline": "RB=F",
    "heating oil": "HO=F",
    "jet fuel": "CL=F",
    "energy": "XLE",

    "gold": "GC=F",
    "silver": "SI=F",
    "copper": "HG=F",
    "platinum": "PL=F",
    "palladium": "PA=F",
    "aluminum": "JJU",
    "steel": "SLX",
    "iron ore": "TIO=F",
    "lithium": "LIT",
    "rare earth": "REMX",
    "uranium": "URA",
    "cobalt": "COBA.L",
    "nickel": "JJN",
    "zinc": "JJZ",

    "corn": "ZC=F",
    "wheat": "ZW=F",
    "soybean": "ZS=F",
    "coffee": "KC=F",
    "sugar": "SB=F",
    "cotton": "CT=F",
    "cocoa": "CC=F",
    "lumber": "LBS=F",
    "cattle": "LE=F",
    "hogs": "HE=F",

    "vix": "^VIX",
    "volatility": "^VIX",
    "fear index": "^VIX",
    "s&p 500": "^GSPC",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow jones": "^DJI",
    "russell 2000": "^RUT",
    "small cap": "^RUT",
    "mid cap": "^MDY",
    "equal weight": "RSP",

    "semiconductor index": "^SOX",
    "philadelphia semiconductor": "^SOX",
    "chip stocks": "^SOX",
    "chip index": "^SOX",
    "technology sector": "XLK",
    "tech sector": "XLK",
    "bank index": "^BKX",
    "financial sector": "XLF",
    "healthcare sector": "XLV",
    "biotech index": "^BTK",
    "biotech sector": "XBI",
    "airline index": "^XAL",
    "transportation index": "^DJT",
    "oil index": "XLE",
    "energy sector": "XLE",
    "utility index": "XLU",
    "utilities sector": "XLU",
    "real estate index": "VNQ",
    "reit index": "VNQ",
    "consumer discretionary": "XLY",
    "consumer staples": "XLP",
    "materials sector": "XLB",
    "industrials sector": "XLI",
    "communication services": "XLC",
    "defensive stocks": "XLP",
    "growth stocks": "VUG",
    "value stocks": "VTV",

    "10 year treasury": "^TNX",
    "10yr yield": "^TNX",
    "2 year treasury": "^IRX",
    "30 year treasury": "^TYX",
    "high yield bond": "HYG",
    "junk bonds": "HYG",
    "investment grade bond": "LQD",
    "corporate bonds": "LQD",
    "tips": "TIP",
    "inflation protected": "TIP",
    "municipal bonds": "MUB",
    "emerging market bonds": "EMB",
    "bond volatility": "^MOVE",

    "dollar index": "DX-Y.NYB",
    "usd index": "DX-Y.NYB",
    "dollar strength": "DX-Y.NYB",
    "euro": "EURUSD=X",
    "yen": "JPY=X",
    "yuan": "CNY=X",
    "pound": "GBPUSD=X",
    "canadian dollar": "CAD=X",
    "australian dollar": "AUD=X",
    "swiss franc": "CHF=X",
    "emerging market currency": "EEM",
    "mexican peso": "MXN=X",
    "korean won": "KRW=X",
    "taiwan dollar": "TWD=X",
    "indian rupee": "INR=X",
    "brazil real": "BRL=X",

    "china": "FXI",
    "china economy": "FXI",
    "china stocks": "FXI",
    "china tech": "CQQQ",
    "taiwan": "EWT",
    "taiwan semiconductor": "EWT",
    "south korea": "EWY",
    "korea semiconductor": "EWY",
    "japan": "EWJ",
    "japan economy": "EWJ",
    "europe": "VGK",
    "eurozone": "VGK",
    "germany": "EWG",
    "uk": "EWU",
    "india": "INDA",
    "brazil": "EWZ",
    "emerging market": "EEM",
    "frontier market": "FM",
    "global market": "VT",
    "world market": "VT",
    "asia pacific": "AAXJ",
    "latin america": "ILF",

    "shipping": "^BDI",
    "baltic dry": "^BDI",
    "global trade": "^BDI",
    "container shipping": "ZIM",
    "freight": "^BDI",
    "supply chain": "^BDI",
    "tanker": "TNK",
    "dry bulk": "SBLK",

    "crypto": "BTC-USD",
    "bitcoin": "BTC-USD",
    "risk appetite": "BTC-USD",

    "market stress": "^VIX",
    "credit stress": "HYG",
    "liquidity": "^VIX",
    "risk off": "GLD",
    "safe haven": "GLD",
    "gold safe haven": "GLD",
}

SCRAPE_CONCEPT_MAP = {

    "air travel demand": "tsa_throughput",
    "airline passengers": "tsa_throughput",
    "passenger traffic": "tsa_throughput",
    "travel demand": "tsa_throughput",
    "airport traffic": "tsa_throughput",
    "domestic travel": "tsa_throughput",
    "passenger volume": "tsa_throughput",
    "tsa": "tsa_throughput",

    "fda approval": "fda_calendar",
    "drug approval": "fda_calendar",
    "regulatory approval": "fda_calendar",
    "clinical trial": "fda_calendar",
    "pdufa": "fda_calendar",
    "drug pipeline": "fda_calendar",
    "biotech catalyst": "fda_calendar",
    "nda approval": "fda_calendar",
    "bla approval": "fda_calendar",
}

def find_data_source(factor_name, data_concept):

    search_text = (factor_name + " " + data_concept).lower()

    best_match = None
    best_score = 0

    for concept, series_id in FRED_CONCEPT_MAP.items():

        concept_words = set(concept.lower().split())
        search_words = set(search_text.split())
        overlap = len(concept_words & search_words)

        score = overlap + (len(concept_words) * 0.1)
        if score > best_score:
            best_score = score
            best_match = ("fred", series_id, concept)

    for concept, ticker_id in YFINANCE_CONCEPT_MAP.items():
        concept_words = set(concept.lower().split())
        search_words = set(search_text.split())
        overlap = len(concept_words & search_words)
        score = overlap + (len(concept_words) * 0.1)
        if score > best_score:
            best_score = score
            best_match = ("yfinance", ticker_id, concept)

    for concept, scrape_id in SCRAPE_CONCEPT_MAP.items():
        concept_words = set(concept.lower().split())
        search_words = set(search_text.split())
        overlap = len(concept_words & search_words)
        score = overlap + (len(concept_words) * 0.1)
        if score > best_score:
            best_score = score
            best_match = ("scrape", scrape_id, concept)

    if best_score >= 1.0:
        return best_match

    return None

def fetch_fred_series(series_id, col_name):

    try:
        fred = Fred(api_key=FRED_KEY)
        data = fred.get_series(
            series_id,
            observation_start=START_DATE,
            observation_end=END_DATE
        )
        df = data.resample("D").interpolate().to_frame(name=col_name)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    except Exception as e:
        print(f"  ✗ FRED {series_id}: {e}")
        return None

def fetch_yfinance_series(ticker_id, col_name):

    try:
        data = yf.download(
            ticker_id,
            start=START_DATE,
            end=END_DATE,
            progress=False
        )
        if data.empty:
            return None
        df = data[["Close"]].copy()
        df.columns = [col_name]
        df.index = pd.to_datetime(df.index).tz_localize(None)

        df[f"{col_name}_change"] = df[col_name].pct_change()
        return df
    except Exception as e:
        print(f"  ✗ yfinance {ticker_id}: {e}")
        return None

def fetch_tsa_throughput():

    print("  Fetching TSA passenger throughput...")
    try:
        url = "https://www.tsa.gov/coronavirus/passenger-throughput"
        tables = pd.read_html(url)
        if not tables:
            return None

        df = tables[0]

        df.columns = [str(c).lower().strip() for c in df.columns]

        date_col = next(
            (c for c in df.columns if "date" in c), None
        )
        throughput_col = next(
            (c for c in df.columns
             if "2024" in c or "2025" in c or "throughput" in c
             or "travelers" in c), None
        )

        if not date_col:
            return None

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None

        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        df["tsa_throughput"] = pd.to_numeric(
            df[numeric_cols[0]].astype(str).str.replace(",", ""),
            errors="coerce"
        )

        df = df.dropna(subset=["date", "tsa_throughput"])
        df = df.set_index("date")[["tsa_throughput"]]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        df = df.resample("D").interpolate()

        print(f"  ✓ TSA throughput: {len(df)} days")
        return df

    except Exception as e:
        print(f"  ✗ TSA scrape failed: {e}")
        return None

def fetch_fda_calendar():

    print("  Fetching FDA approval calendar...")
    try:
        url = "https://www.fda.gov/patients/drug-approval-process/novel-drug-approvals-fda"
        tables = pd.read_html(url)
        if not tables:
            return None

        df = tables[0]
        df.columns = [str(c).lower().strip() for c in df.columns]

        date_col = next(
            (c for c in df.columns
             if "date" in c or "approv" in c), None
        )
        if not date_col:
            return None

        df["approval_date"] = pd.to_datetime(
            df[date_col], errors="coerce"
        )
        df = df.dropna(subset=["approval_date"])

        date_range = pd.date_range(
            start=START_DATE, end=END_DATE, freq="D"
        )
        fda_df = pd.DataFrame(index=date_range)
        fda_df["fda_approval_30d"] = 0

        for approval_date in df["approval_date"]:
            window_start = approval_date - timedelta(days=30)
            window_end = approval_date
            mask = (fda_df.index >= window_start) & \
                   (fda_df.index <= window_end)
            fda_df.loc[mask, "fda_approval_30d"] = 1

        fda_df.index = pd.to_datetime(fda_df.index).tz_localize(None)
        print(f"  ✓ FDA calendar: {len(df)} approvals found")
        return fda_df

    except Exception as e:
        print(f"  ✗ FDA scrape failed: {e}")
        return None

def get_industry_signals(ticker, company_info):

    print(f"\n[Industry Signals] Building AI-driven signal set for {ticker}...")

    factors_json = get_driving_factors(ticker, company_info)

    if not factors_json:
        print("  [AI] Could not get factors — skipping industry signals")
        return pd.DataFrame()

    print(f"\n  Mapping {len(factors_json['factors'])} factors to data sources...")

    data_pulls = []
    seen_sources = set()

    for factor in factors_json["factors"]:
        factor_name = factor["factor"]
        data_concept = factor.get("data_concept", "")

        match = find_data_source(factor_name, data_concept)

        if match:
            source_type, source_id, matched_concept = match

            if source_id not in seen_sources:
                seen_sources.add(source_id)

                col_name = (
                    factor_name.lower()
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("-", "_")
                    .replace("&", "and")
                    [:25]
                )
                data_pulls.append((col_name, source_type, source_id))
                print(f"  ✓ '{factor_name}' → {source_type}:{source_id} "
                      f"(via '{matched_concept}')")
        else:
            print(f"  ✗ '{factor_name}' → no data source found")

    if not data_pulls:
        print("  No data sources mapped — skipping industry signals")
        return pd.DataFrame()

    print(f"\n  Pulling {len(data_pulls)} data sources...")
    all_dfs = []

    for col_name, source_type, source_id in data_pulls:
        if source_type == "fred":
            df = fetch_fred_series(source_id, col_name)
            if df is not None:
                all_dfs.append(df)
                print(f"  ✓ {col_name} (FRED: {source_id})")

        elif source_type == "yfinance":
            df = fetch_yfinance_series(source_id, col_name)
            if df is not None:
                all_dfs.append(df)
                print(f"  ✓ {col_name} (yfinance: {source_id})")

        elif source_type == "scrape":
            if source_id == "tsa_throughput":
                df = fetch_tsa_throughput()
                if df is not None:
                    all_dfs.append(df)
            elif source_id == "fda_calendar":
                df = fetch_fda_calendar()
                if df is not None:
                    all_dfs.append(df)

    if not all_dfs:
        print("  No data successfully pulled")
        return pd.DataFrame()

    combined = all_dfs[0]
    for df in all_dfs[1:]:
        combined = combined.join(df, how="outer")

    combined = combined.ffill().bfill()

    combined.columns = [f"ind_{col}" for col in combined.columns]
    combined.index = pd.to_datetime(combined.index).tz_localize(None)

    combined = combined[
        (combined.index >= START_DATE) &
        (combined.index <= END_DATE)
    ]

    print(f"\n  Industry signals complete:")
    print(f"  {len(combined.columns)} signals pulled")
    print(f"  {len(combined)} days of data")
    print(f"  Columns: {list(combined.columns)}")

    return combined
