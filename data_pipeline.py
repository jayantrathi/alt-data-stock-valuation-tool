import os
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fredapi import Fred
from pytrends.request import TrendReq
from industry_signals import get_industry_signals
warnings.filterwarnings("ignore")

load_dotenv()


START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
FRED_KEY = os.getenv("FRED_API_KEY")


def get_price_data(ticker):
    print(f"\n[1/6] Fetching price data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(start=START_DATE, end=END_DATE)
    df = df[["Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.columns = ["price", "volume"]


    df["return_1w"] = df["price"].pct_change(5).shift(-5)
    df["return_2w"] = df["price"].pct_change(10).shift(-10)
    df["return_1m"] = df["price"].pct_change(21).shift(-21)
    df["return_3m"] = df["price"].pct_change(63).shift(-63)

    print(f"  Got {len(df)} days of price data")
    return df


def get_macro_data():
    print("\n[2/6] Fetching macro data from FRED...")
    fred = Fred(api_key=FRED_KEY)


    series = {
        "fed_rate":      "FEDFUNDS",
        "inflation":     "CPIAUCSL",
        "unemployment":  "UNRATE",
        "vix":           "VIXCLS",
        "usd_index":     "DTWEXBGS",
        "treasury_10y":  "DGS10",
        "consumer_conf": "UMCSENT",
        "oil":           "DCOILWTICO",
    }

    macro_df = pd.DataFrame()

    for name, series_id in series.items():
        try:
            data = fred.get_series(
                series_id,
                observation_start=START_DATE,
                observation_end=END_DATE
            )
            data = data.resample("D").interpolate()
            macro_df[name] = data
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")

    macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)
    return macro_df


def get_sector_data(ticker):
    print("\n[3/6] Fetching sector ETF data...")


    stock = yf.Ticker(ticker)
    sector = stock.info.get("sector", "Technology")

    sector_etfs = {
        "Technology":            "XLK",
        "Healthcare":            "XLV",
        "Financial Services":    "XLF",
        "Consumer Cyclical":     "XLY",
        "Consumer Defensive":    "XLP",
        "Industrials":           "XLI",
        "Energy":                "XLE",
        "Utilities":             "XLU",
        "Real Estate":           "XLRE",
        "Basic Materials":       "XLB",
        "Communication Services":"XLC",
    }

    etf = sector_etfs.get(sector, "SPY")
    print(f"  Sector: {sector} → ETF: {etf}")

    sector_df = pd.DataFrame()


    etf_data = yf.download(etf, start=START_DATE, end=END_DATE, progress=False)
    sector_df["sector_etf"] = etf_data["Close"]


    spy_data = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False)
    sector_df["sp500"] = spy_data["Close"]


    sector_df["sector_return"] = sector_df["sector_etf"].pct_change()
    sector_df["market_return"] = sector_df["sp500"].pct_change()

    sector_df.index = pd.to_datetime(sector_df.index).tz_localize(None)
    print(f"  ✓ sector ETF and S&P 500")
    return sector_df, etf


def get_trends_data(ticker, company_name):
    print("\n[4/6] Fetching Google Trends data...")

    company_short = company_name.split()[0].strip(".,")

    pytrends = TrendReq(hl="en-US", tz=360)

    trends_df = pd.DataFrame()


    search_terms = [
        company_short,
        f"{ticker} stock",
        f"{company_short} stock price",
    ]

    for term in search_terms:
        try:
            pytrends.build_payload(
                [term],
                timeframe=f"{START_DATE} {END_DATE}",
                geo="US"
            )
            data = pytrends.interest_over_time()
            if not data.empty:
                col_name = term.replace(" ", "_").lower()
                trends_df[col_name] = data[term]
                print(f"  ✓ '{term}'")
        except Exception as e:
            print(f"  ✗ '{term}' failed: {e}")

    if not trends_df.empty:
        trends_df.index = pd.to_datetime(trends_df.index).tz_localize(None)

        trends_df = trends_df.resample("D").interpolate()

    return trends_df


def get_competitor_data(ticker):
    print("\n[5/6] Fetching competitor data...")

    stock = yf.Ticker(ticker)
    info = stock.info
    sector = info.get("sector", "Technology")
    industry = info.get("industry", "")

    print(f"  Sector:   {sector}")
    print(f"  Industry: {industry}")


    industry_fallback = {


        "Consumer Electronics": [
            "AAPL", "SONY", "SSNLF", "HPQ", "DELL", "LOGI", "HEAR",
            "VZIO", "ROKU", "GPRO", "POLA", "IRBT", "VOXX", "KOSS"
        ],
        "Semiconductors": [
            "NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT",
            "LRCX", "KLAC", "TSM", "ASML", "MRVL", "ON", "SWKS", "QRVO",
            "MPWR", "WOLF", "ACLS", "COHU", "ONTO", "FORM", "AMBA", "SLAB"
        ],
        "Semiconductor Equipment": [
            "AMAT", "LRCX", "KLAC", "ASML", "TER", "ACLS", "ONTO",
            "COHU", "FORM", "UCTT", "ICHR", "NVMI", "CAMT", "KLIC"
        ],
        "Software—Application": [
            "MSFT", "ORCL", "SAP", "CRM", "ADBE", "NOW", "INTU", "WDAY",
            "SNOW", "PLTR", "DDOG", "ZM", "DOCU", "HUBS", "VEEV", "BILL",
            "SMAR", "COUP", "APPF", "PCTY", "PAYC", "JAMF", "FROG", "GTLB"
        ],
        "Software—Infrastructure": [
            "MSFT", "IBM", "ORCL", "VMW", "PANW", "CRWD", "ZS", "OKTA",
            "NET", "FTNT", "CYBR", "S", "TENB", "RPD", "VRNS", "QLYS",
            "SAIL", "DOMO", "MDB", "ESTC", "SUMO", "NEWR", "DT", "FSLY"
        ],
        "Internet Content & Information": [
            "GOOGL", "META", "SNAP", "PINS", "TWTR", "YELP", "IAC",
            "ZG", "MTCH", "BMBL", "ANGI", "ELF", "OPEN", "CARS", "TDC"
        ],
        "Internet Retail": [
            "AMZN", "EBAY", "ETSY", "SHOP", "W", "CHWY", "OSTK",
            "FTCH", "REAL", "POSH", "WISH", "PETS", "PRTS", "FLXS"
        ],
        "IT Services": [
            "IBM", "ACN", "INFY", "WIT", "CTSH", "DXC", "EPAM",
            "GLOB", "LDOS", "SAIC", "CACI", "BAH", "EXLS", "KFRC"
        ],
        "Computer Hardware": [
            "AAPL", "DELL", "HPQ", "HPE", "NTAP", "PSTG", "WDC",
            "STX", "SMCI", "NTNX", "PEGA", "SANM", "PLXS", "FLEX"
        ],
        "Electronic Components": [
            "TEL", "APH", "GLW", "MXIM", "ADI", "NXPI", "MCHP",
            "KEYS", "ITRI", "BEL", "CTS", "VICR", "PLPC", "BEL"
        ],
        "Electronics & Computer Distribution": [
            "ARW", "AVT", "SNX", "TD", "SCSC", "NSIT", "CNXN",
            "CLFD", "DSGN", "IIIV", "PLAB", "LYTS", "NTIC", "PCYC"
        ],


        "Telecom Services": [
            "T", "VZ", "TMUS", "CMCSA", "CHTR", "LUMN", "TDS",
            "USM", "SHEN", "CABO", "WOW", "CNSL", "OOMA", "LMND"
        ],
        "Entertainment": [
            "DIS", "NFLX", "PARA", "WBD", "FOXA", "LGF-A", "AMC",
            "IMAX", "CNK", "MCS", "MKGI", "MANU", "WWE", "EDR"
        ],
        "Broadcasting": [
            "DIS", "FOXA", "PARA", "WBD", "NXST", "SBGI", "GTN",
            "GCI", "NYT", "MDP", "SSP", "AMCX", "VIAC", "IHRT"
        ],
        "Electronic Gaming & Multimedia": [
            "ATVI", "EA", "TTWO", "RBLX", "U", "DKNG", "PENN",
            "SKLZ", "GENI", "HUYA", "DOYU", "NTES", "BILI", "SE"
        ],
        "Publishing": [
            "NYT", "GCI", "MDP", "SSP", "NWSA", "NWS", "SCHL",
            "HMHC", "WBGO", "EDUC", "MKTX", "INFO", "DFIN", "VVNT"
        ],


        "Airlines": [
            "DAL", "UAL", "AAL", "ALK", "JBLU", "SAVE", "HA",
            "ULCC", "RYAAY", "WIZZ", "IAG", "ICAD", "MESA", "SKYW"
        ],
        "Airports & Air Services": [
            "RYAAY", "DAL", "UAL", "AAL", "ALK", "JBLU", "HA",
            "AAWW", "ATSG", "CLXT", "WAIR", "AVAV", "RAVN", "FLYA"
        ],
        "Aerospace & Defense": [
            "BA", "LMT", "RTX", "NOC", "GD", "HII", "TDG", "KTOS",
            "AJRD", "LDOS", "CACI", "BAH", "HEI", "TGI", "MOOG",
            "CW", "DRS", "FLIR", "VSE", "BWXT", "AXON", "AVAV"
        ],
        "Railroads": [
            "UNP", "CSX", "NSC", "CP", "CN", "WAB", "TRN",
            "GBX", "RAIL", "ARII", "HFBL", "GATX", "RXO", "EXPO"
        ],
        "Trucking": [
            "UPS", "FDX", "ODFL", "SAIA", "XPO", "JBHT", "CHRW",
            "WERN", "KNX", "LSTR", "HTLD", "MRTN", "USX", "PTSI"
        ],
        "Marine Shipping": [
            "ZIM", "DAC", "GSL", "SFL", "GOGL", "EGLE", "SBLK",
            "SALT", "TOPS", "PRGN", "CTRM", "SHIP", "GASS", "SINO"
        ],
        "Integrated Freight & Logistics": [
            "UPS", "FDX", "CHRW", "EXPD", "XPO", "DHLGY", "TNT",
            "GXO", "RXO", "HUBG", "ECHO", "FWRD", "ATRI", "UHAL"
        ],
        "Farm & Heavy Construction Machinery": [
            "DE", "CAT", "AGCO", "CNH", "PCAR", "OSK", "TEX",
            "CNHI", "TITN", "ASTE", "HLIO", "LNN", "ALAMO", "SHYF"
        ],
        "Specialty Industrial Machinery": [
            "HON", "EMR", "ITW", "PH", "ROK", "AME", "ROP",
            "FTV", "GNRC", "XYL", "XYLD", "FLOW", "CSWI", "GTLS"
        ],
        "Engineering & Construction": [
            "FLR", "KBR", "J", "STRL", "MTZ", "PWR", "PRIM",
            "TTEK", "ARCB", "MYRG", "ROAD", "IESC", "GLDD", "CATO"
        ],
        "Conglomerates": [
            "GE", "HON", "MMM", "EMR", "ITW", "DOV", "ROP",
            "SPXC", "AMETEK", "ACCO", "HLF", "SEB", "TNC", "CODI"
        ],
        "Waste Management": [
            "WM", "RSG", "WCN", "CWST", "SRCL", "US", "ADSW",
            "HCCI", "MEG", "NVRI", "CEVA", "PKOH", "CLH", "ARIS"
        ],
        "Security & Protection Services": [
            "AXON", "ADT", "ALLE", "BCO", "BRINKS", "PRSO", "MSA",
            "REZI", "NSSC", "NAPCO", "DGLY", "IDSY", "ISNS", "IPIX"
        ],
        "Staffing & Employment Services": [
            "MAN", "ADP", "PAYX", "RHI", "KFRC", "KELYA", "HURN",
            "HSII", "MPS", "HIRE", "CCRN", "TBI", "NRCG", "JOBS"
        ],
        "Rental & Leasing Services": [
            "URI", "RSC", "AL", "GATX", "TRTN", "CAR", "HTZ",
            "RADI", "MGRC", "WLFC", "NACCO", "GFN", "HCCI", "BFAM"
        ],

        "Auto Manufacturers": [
            "TSLA", "F", "GM", "TM", "HMC", "STLA", "RIVN",
            "LCID", "NIO", "LI", "XPEV", "BYDDY", "FFIE", "FSR"
        ],
        "Auto Parts": [
            "MGA", "BWA", "LEA", "APTV", "DAN", "MOD", "CPS",
            "THRM", "DORM", "SMP", "STRZA", "GT", "SUP", "FOXF"
        ],
        "Auto & Truck Dealerships": [
            "AN", "PAG", "LAD", "KMX", "SAH", "ABG", "GPI",
            "CVNA", "CARG", "TRUE", "RUOFF", "VROOM", "AUTO", "CDK"
        ],
        "Restaurants": [
            "MCD", "SBUX", "CMG", "YUM", "QSR", "DPZ", "WEN",
            "JACK", "SHAK", "TXRH", "DIN", "DENN", "CAKE", "EAT"
        ],
        "Hotels & Motels": [
            "MAR", "HLT", "IHG", "H", "WH", "CHH", "RHP",
            "PK", "SHO", "APLE", "CLDT", "CHATM", "STAY", "VCNX"
        ],
        "Resorts & Casinos": [
            "LVS", "MGM", "WYNN", "CZR", "MLCO", "BYD", "PEN",
            "GDEN", "FULL", "EVRI", "SGMS", "AGS", "DKNG", "ACCD"
        ],
        "Travel Services": [
            "BKNG", "EXPE", "TRIP", "ABNB", "LYFT", "UBER", "DESP",
            "TRVG", "MMYT", "SEERA", "LMND", "HWAY", "EDR", "TZOO"
        ],
        "Specialty Retail": [
            "HD", "LOW", "TGT", "COST", "WMT", "DG", "DLTR",
            "BBY", "WSM", "RH", "BBBY", "GME", "FIVE", "OLLI"
        ],
        "Apparel Retail": [
            "NKE", "LULU", "UAA", "PVH", "RL", "HBI", "VFC",
            "SKX", "DECK", "CROX", "BOOT", "GOOS", "ONON", "COLM"
        ],
        "Apparel Manufacturing": [
            "NKE", "PVH", "RL", "HBI", "VFC", "UA", "GOOS",
            "CPRI", "MOV", "FL", "SHOO", "DKNY", "WRNR", "OXM"
        ],
        "Footwear & Accessories": [
            "NKE", "SKX", "DECK", "CROX", "BOOT", "ONON", "SHOO",
            "FOSL", "MOV", "GRMN", "KORS", "TIF", "SIG", "ZUMZ"
        ],
        "Home Improvement Retail": [
            "HD", "LOW", "FND", "FLOR", "TTS", "FBHS", "JELD",
            "MHK", "AWI", "TILE", "TREX", "AZEK", "BECN", "GMS"
        ],
        "Furnishings, Fixtures & Appliances": [
            "WHR", "ETH", "MLKN", "SNBR", "PRPL", "LESL",
            "LOVE", "HOFT", "FLXS", "KIRK", "BBBY", "WSM", "RH"
        ],
        "Gambling": [
            "DKNG", "PENN", "MGM", "CZR", "LVS", "WYNN", "BYD",
            "RSI", "GDEN", "AGS", "SGMS", "EVRI", "BALY", "GAN"
        ],
        "Leisure": [
            "PTON", "NCLH", "CCL", "RCL", "HAS", "MAT", "PLNT",
            "COLM", "VSTO", "HZNP", "GRMN", "MGA", "MODV", "NAUT"
        ],
        "Packaging & Containers": [
            "IP", "PKG", "SEE", "SON", "GPK", "BERY", "SLGN",
            "ATR", "AEY", "UFPI", "MERC", "PTVE", "SILGA", "TRS"
        ],
        "Personal Services": [
            "SCI", "CSV", "ROL", "REVG", "ACCO", "HRB",
            "EFC", "PRSC", "PAYO", "RELY", "PAYC", "BFAM"
        ],

        "Beverages—Non-Alcoholic": [
            "KO", "PEP", "MNST", "KDP", "CELH", "FIZZ",
            "COTT", "NRGV", "REED", "WTER", "NOMD", "COKE"
        ],
        "Beverages—Alcoholic": [
            "BUD", "TAP", "STZ", "SAM", "ABEV", "DEO",
            "BF-B", "MGPI", "WEST", "HOOK", "EAST", "CRAFT"
        ],
        "Beverages—Wineries & Distilleries": [
            "STZ", "BF-B", "MGPI", "DEO", "ABEV", "SAM",
            "HOOK", "WVVI", "EAST", "CASK", "BCLI", "MEAD"
        ],
        "Grocery Stores": [
            "KR", "WMT", "COST", "SFM", "GO", "VLGEA",
            "IMKTA", "WINN", "CHEF", "PFGC", "USFD", "SPTN"
        ],
        "Household & Personal Products": [
            "PG", "CL", "KMB", "CHD", "ENR", "EPC",
            "SPB", "COTY", "CENT", "HBB", "ACCO", "RCKY"
        ],
        "Packaged Foods": [
            "GIS", "K", "CPB", "SJM", "CAG", "HSY", "MKC",
            "MDLZ", "HRL", "LW", "NOMD", "SMPL", "JJSF", "FRPT"
        ],
        "Tobacco": [
            "PM", "MO", "BTI", "LO", "IGM", "SWMAY",
            "TPB", "IMBBY", "VGR", "SSTK", "XXII", "GNLN"
        ],
        "Farm Products": [
            "ADM", "BG", "INGR", "MOS", "CF", "IPI",
            "CTVA", "FMC", "SMG", "ANDE", "CALM", "VITL"
        ],
        "Food Distribution": [
            "USFD", "PFGC", "SPTN", "SYY", "CHEF", "UNFI",
            "CORE", "NTST", "BRBR", "STKL", "JJSF", "FRPT"
        ],
        "Discount Stores": [
            "WMT", "TGT", "COST", "DG", "DLTR", "BIG",
            "FIVE", "OLLI", "BURL", "TJX", "ROSS", "TUES"
        ],
        "Drug Stores": [
            "CVS", "WBA", "RAD", "HIBB", "PDCO", "HSIC",
            "PETS", "CHWY", "PRGO", "PAHC", "PNTM", "PHAR"
        ],

        "Drug Manufacturers—General": [
            "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY",
            "AZN", "NVS", "RHHBY", "SNY", "GSK", "TAK"
        ],
        "Drug Manufacturers—Specialty & Generic": [
            "TEVA", "MYL", "PRGO", "ENDP", "AGN", "JAZZ",
            "SUPN", "LNTH", "AMAG", "AKRX", "HZNP", "ITCI"
        ],
        "Biotechnology": [
            "AMGN", "GILD", "BIIB", "REGN", "VRTX", "MRNA",
            "BNTX", "SGEN", "ALNY", "INCY", "EXEL", "RARE",
            "IONS", "SRPT", "BLUE", "BMRN", "ARVN", "RCUS"
        ],
        "Medical Devices": [
            "MDT", "ABT", "SYK", "BSX", "EW", "ZBH",
            "BDX", "BAX", "ISRG", "HOLX", "VAR", "NUS",
            "SWAV", "PODD", "TNDM", "DXCM", "IART", "NVCR"
        ],
        "Medical Instruments & Supplies": [
            "BDX", "BAX", "COO", "HSIC", "PDCO", "CTLT",
            "MMSI", "ICAD", "ANGO", "LMAT", "ATRC", "CNMD"
        ],
        "Diagnostics & Research": [
            "TMO", "DHR", "IQV", "A", "BIO", "ILMN",
            "EXAS", "NTRA", "SDGR", "ONEM", "PSNL", "CDNA"
        ],
        "Health Information Services": [
            "UNH", "CVS", "CI", "HUM", "MOH", "CNC",
            "ELV", "HCA", "THC", "UHS", "OSCR", "CLOV"
        ],
        "Healthcare Plans": [
            "UNH", "CVS", "CI", "HUM", "MOH", "CNC",
            "ELV", "OSCR", "CLOV", "BHVN", "ALHC", "ACCD"
        ],
        "Medical Care Facilities": [
            "HCA", "THC", "UHS", "CYH", "ACHC", "SGRY",
            "NVST", "AMSF", "ADUS", "AMED", "LHCG", "ENSG"
        ],
        "Pharmaceutical Retailers": [
            "CVS", "WBA", "RAD", "PRGO", "PDCO", "HSIC",
            "PETS", "CHWY", "PAHC", "PNTM", "PHAR", "HCAT"
        ],

        "Banks—Diversified": [
            "JPM", "BAC", "WFC", "C", "USB", "PNC",
            "TFC", "FITB", "KEY", "CFG", "HBAN", "RF"
        ],
        "Banks—Regional": [
            "USB", "PNC", "TFC", "FITB", "KEY", "CFG",
            "HBAN", "RF", "MTB", "ZION", "CMA", "WTFC",
            "FHN", "BOH", "BOKF", "FFIN", "CVBF", "CATY"
        ],
        "Asset Management": [
            "BLK", "SCHW", "MS", "GS", "BX", "APO",
            "KKR", "CG", "ARES", "BAM", "OWL", "BLUE"
        ],
        "Capital Markets": [
            "GS", "MS", "JPM", "BAC", "UBS", "DB",
            "RJF", "SF", "LPLA", "MKTX", "ICE", "CME"
        ],
        "Insurance—Diversified": [
            "BRK-B", "MET", "PRU", "AIG", "AFL", "ALL",
            "PGR", "TRV", "CB", "HIG", "UNM", "GL"
        ],
        "Insurance—Property & Casualty": [
            "PGR", "ALL", "TRV", "CB", "HIG", "CNA",
            "WRB", "CINF", "SIGI", "ERIE", "UFG", "KMPR"
        ],
        "Insurance—Life": [
            "MET", "PRU", "AFL", "LNC", "UNM", "GL",
            "PFG", "FG", "CNO", "NWLI", "SAMPO", "ERIE"
        ],
        "Insurance—Specialty": [
            "RLI", "WRB", "STFC", "SIGI", "KMPR", "ERIE",
            "HWNI", "DGICA", "ACGL", "RNR", "AWH", "GBLI"
        ],
        "Credit Services": [
            "V", "MA", "AXP", "DFS", "SYF", "COF",
            "ALLY", "OMF", "CACC", "PRAA", "ENVA", "QFIN"
        ],
        "Financial Data & Stock Exchanges": [
            "ICE", "CME", "NDAQ", "CBOE", "MSCI", "SPGI",
            "MCO", "FDS", "MKTX", "TW", "BLND", "OPEN"
        ],
        "Mortgage Finance": [
            "FNM", "FRE", "RKT", "UWMC", "GHLD", "PFSI",
            "WAL", "NRZ", "MITT", "TWO", "BXMT", "KREF"
        ],

        "Oil & Gas E&P": [
            "XOM", "CVX", "COP", "EOG", "PXD", "DVN",
            "MRO", "APA", "HES", "OXY", "FANG", "SM",
            "CDEV", "WPX", "CPE", "PDCE", "ESTE", "BATL"
        ],
        "Oil & Gas Integrated": [
            "XOM", "CVX", "BP", "SHEL", "TTE", "ENB",
            "SU", "IMO", "CNQ", "CVE", "OVV", "TRGP"
        ],
        "Oil & Gas Midstream": [
            "EPD", "ET", "MPLX", "WES", "TRGP", "OKE",
            "KMI", "LNG", "CQP", "DCP", "NGL", "SMLP"
        ],
        "Oil & Gas Refining & Marketing": [
            "MPC", "PSX", "VLO", "HFC", "PBF", "DKL",
            "CLMT", "PARR", "DINO", "CVRR", "RRMS", "CALUMET"
        ],
        "Oil & Gas Equipment & Services": [
            "SLB", "HAL", "BKR", "FTI", "RES", "LBRT",
            "NES", "ACDC", "KLXE", "OIS", "PUMP", "NINE"
        ],
        "Coal": [
            "BTU", "ARCH", "AMR", "CEIX", "ARLP", "CONSOL",
            "NRP", "SXC", "RHINO", "FELP", "METC", "HNRG"
        ],
        "Uranium": [
            "CCJ", "UEC", "UUUU", "DNN", "URG", "BQSSF",
            "FCU", "NXE", "PDN", "PEN", "SPUT", "URNM"
        ],
        "Solar": [
            "ENPH", "SEDG", "FSLR", "RUN", "SPWR", "CSIQ",
            "JKS", "ARRY", "NOVA", "SHLS", "MAXN", "POLA"
        ],
        "Oil & Gas Drilling": [
            "VAL", "RIG", "NE", "DO", "HP", "PTEN",
            "KLXE", "OIS", "PUMP", "NINE", "ACDC", "WTTR"
        ],

        "Specialty Chemicals": [
            "LIN", "APD", "SHW", "ECL", "PPG", "RPM",
            "IFF", "ALB", "AVNT", "OLIN", "EMN", "CE",
            "HUN", "TROX", "CC", "GTHX", "ASIX", "KWR"
        ],
        "Agricultural Inputs": [
            "MOS", "CF", "NTR", "IPI", "SMG", "CTVA",
            "FMC", "ANDE", "GNSS", "YARA", "ICL", "SQM"
        ],
        "Gold": [
            "NEM", "GOLD", "AEM", "KGC", "AGI", "EGO",
            "AU", "HMY", "BTG", "OR", "WPM", "FNV"
        ],
        "Silver": [
            "WPM", "PAAS", "AG", "CDE", "HL", "FSM",
            "MAG", "SILV", "GATO", "IPX", "MTA", "ADY"
        ],
        "Copper": [
            "FCX", "SCCO", "HBM", "TGB", "ERO", "CMMC",
            "FM", "NOVR", "CPER", "CU", "COPX", "GBXG"
        ],
        "Steel": [
            "NUE", "STLD", "CLF", "X", "CMC", "RS",
            "GGB", "MT", "PKX", "TMST", "ZEUS", "HYLLY"
        ],
        "Aluminum": [
            "AA", "CENX", "KALU", "ARNC", "ALCOA", "NHYDY",
            "RIO", "BHP", "VALE", "ACH", "CSTM", "NOVR"
        ],
        "Lumber & Wood Production": [
            "WY", "RFP", "PCH", "JAMF", "LPX", "UFP",
            "UFPI", "SWM", "CLW", "MERC", "KW", "WOOD"
        ],
        "Paper & Paper Products": [
            "IP", "PKG", "SEE", "SON", "GPK", "BERY",
            "ATR", "SWM", "CLW", "MERC", "SON", "SYLVAMO"
        ],
        "Mining": [
            "RIO", "BHP", "VALE", "FCX", "NEM", "GOLD",
            "AA", "X", "NUE", "CLF", "SCCO", "MP"
        ],

        "REIT—Retail": [
            "SPG", "O", "NNN", "KIM", "REG", "BRX",
            "WRI", "MAC", "CBL", "SKT", "RPT", "UE"
        ],
        "REIT—Residential": [
            "EQR", "AVB", "ESS", "MAA", "UDR", "CPT",
            "NHI", "INVH", "AMH", "TRICON", "NRZ", "AIR"
        ],
        "REIT—Office": [
            "BXP", "VNO", "SLG", "HIW", "DEA", "PDM",
            "CXP", "PGRE", "ESRT", "OFC", "CLI", "PKY"
        ],
        "REIT—Industrial": [
            "PLD", "DRE", "FR", "EGP", "STAG", "REXR",
            "LXP", "TRNO", "ILPT", "IIPR", "GTY", "MFTB"
        ],
        "REIT—Healthcare Facilities": [
            "WELL", "VTR", "OHI", "NHI", "HR", "DOC",
            "CHCT", "LTC", "SBRA", "SNH", "UHT", "GMRE"
        ],
        "REIT—Hotel & Motel": [
            "HST", "RHP", "PK", "SHO", "APLE", "CLDT",
            "CHATM", "STAY", "INN", "XHR", "VCNX", "RLJ"
        ],
        "REIT—Specialty": [
            "AMT", "CCI", "EQIX", "DLR", "SBA", "CONE",
            "QTS", "UNIT", "LMND", "IRM", "SAFE", "MODIV"
        ],
        "REIT—Diversified": [
            "WPC", "VICI", "GLPI", "EPRT", "FCPT", "NTST",
            "PINE", "GIPR", "PLYM", "IIPR", "GTY", "BRT"
        ],
        "Real Estate Services": [
            "CBRE", "JLL", "CWK", "MMI", "RMR", "FSV",
            "HFF", "RDFN", "OPEN", "Z", "ZG", "RMAX"
        ],
        "Real Estate—Development": [
            "TOL", "DHI", "LEN", "PHM", "NVR", "MDC",
            "TMHC", "LGIH", "MHO", "TPH", "BZH", "SKY"
        ],

        "Utilities—Regulated Electric": [
            "NEE", "DUK", "SO", "D", "AEP", "EXC",
            "SRE", "XEL", "ED", "WEC", "ES", "ETR",
            "FE", "CMS", "CNP", "AEE", "LNT", "EVRG"
        ],
        "Utilities—Regulated Gas": [
            "SRE", "NI", "ATO", "SW", "OGE", "NWE",
            "SPKE", "RGCO", "LAIX", "SJI", "NWNG", "CWCO"
        ],
        "Utilities—Regulated Water": [
            "AWK", "WTR", "CWT", "MSEX", "YORW", "CTWS",
            "SJW", "GWRS", "CWCO", "ARTNA", "MSWC", "PCYO"
        ],
        "Utilities—Renewable": [
            "NEE", "ENPH", "SEDG", "FSLR", "RUN", "SPWR",
            "BEP", "CWEN", "AY", "HASI", "NOVA", "VVPR"
        ],
        "Utilities—Independent Power Producers": [
            "VST", "NRG", "AES", "GEN", "CLNE", "MNTK",
            "AMRC", "CDPQ", "SPKE", "MPLX", "ETR", "PEG"
        ],
    }

    print("\n  Building dynamic candidate universe...")
    candidates = []


    try:
        sp500 = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            storage_options={"User-Agent": "Mozilla/5.0"}
        )[0]
        sp500_tickers = sp500["Symbol"].str.replace(".", "-").tolist()
        candidates.extend(sp500_tickers)
        print(f"  S&P 500 Wikipedia: {len(sp500_tickers)} tickers")
    except Exception as e:
        print(f"  Wikipedia failed: {e}")


    all_etfs = ["XLK", "XLV", "XLF", "XLY", "XLP",
                "XLI", "XLE", "XLU", "XLRE", "XLB", "XLC"]
    etf_count = 0
    for etf_sym in all_etfs:
        try:
            etf_t = yf.Ticker(etf_sym)
            etf_data = etf_t.funds_data
            if etf_data is not None:
                holdings = etf_data.top_holdings
                if holdings is not None and not holdings.empty:
                    candidates.extend(holdings.index.tolist())
                    etf_count += len(holdings)
        except:
            continue
    print(f"  Sector ETFs: {etf_count} tickers added")


    try:
        iwb = yf.Ticker("IWB")
        iwb_data = iwb.funds_data
        if iwb_data is not None:
            holdings = iwb_data.top_holdings
            if holdings is not None and not holdings.empty:
                candidates.extend(holdings.index.tolist())
                print(f"  Russell 1000: {len(holdings)} tickers added")
    except Exception as e:
        print(f"  Russell 1000 failed: {e}")


    candidates = list(dict.fromkeys(
        [t for t in candidates if t != ticker]
    ))
    print(f"  Total unique candidates: {len(candidates)}")


    print(f"\n  Screening for industry: '{industry}'...")
    industry_matches = []
    sector_matches = []

    for candidate in candidates[:100]:
        try:
            c_info = yf.Ticker(candidate).info
            c_industry = c_info.get("industry", "")
            c_sector = c_info.get("sector", "")
            if c_industry == industry:
                industry_matches.append(candidate)
            elif c_sector == sector:
                sector_matches.append(candidate)
        except:
            continue

    print(f"  Exact industry matches: {len(industry_matches)}")
    print(f"  Sector matches:         {len(sector_matches)}")


    peer_pool = industry_matches if len(industry_matches) >= 3 \
        else industry_matches + sector_matches
    print(f"  Peer pool size:         {len(peer_pool)}")


    print(f"\n  Ranking by price correlation...")
    target_data = yf.download(
        ticker, start=START_DATE, end=END_DATE, progress=False
    )
    target_returns = target_data["Close"].pct_change().dropna()

    def rank_by_correlation(pool):
        correlations = {}
        for peer in pool:
            try:
                peer_data = yf.download(
                    peer, start=START_DATE,
                    end=END_DATE, progress=False
                )
                if peer_data.empty:
                    continue
                peer_returns = peer_data["Close"].pct_change().dropna()
                aligned = pd.concat(
                    [target_returns, peer_returns], axis=1
                ).dropna()
                if len(aligned) < 100:
                    continue
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                correlations[peer] = abs(corr)
            except:
                continue
        return sorted(
            correlations.items(), key=lambda x: x[1], reverse=True
        )

    ranked = rank_by_correlation(peer_pool)


    CORRELATION_THRESHOLD = 0.5
    top5_corrs = [corr for _, corr in ranked[:5]]
    avg_corr = np.mean(top5_corrs) if top5_corrs else 0

    print(f"\n  Top peers from dynamic system:")
    for peer, corr in ranked[:8]:
        try:
            p_industry = yf.Ticker(peer).info.get("industry", "unknown")
        except:
            p_industry = "unknown"
        print(f"    {peer}: {corr:.3f}  ({p_industry})")

    print(f"\n  Average correlation of top 5: {avg_corr:.3f}")
    print(f"  Quality threshold: {CORRELATION_THRESHOLD}")

    fallback_triggered = False
    if avg_corr < CORRELATION_THRESHOLD:
        print(f"\n  ⚠ Quality gate failed — triggering industry fallback preset...")
        fallback_triggered = True


        fallback_tickers = industry_fallback.get(industry, [])


        if not fallback_tickers:
            for key in industry_fallback:
                if any(word in industry.lower()
                       for word in key.lower().split()):
                    fallback_tickers = industry_fallback[key]
                    print(f"  Matched fallback key: '{key}'")
                    break

        if fallback_tickers:

            fallback_tickers = [t for t in fallback_tickers
                                 if t != ticker]
            print(f"  Fallback pool: {fallback_tickers}")


            expanded_pool = list(dict.fromkeys(
                fallback_tickers + peer_pool
            ))


            print(f"  Re-ranking {len(expanded_pool)} candidates...")
            ranked = rank_by_correlation(expanded_pool)

            print(f"\n  Top peers after fallback expansion:")
            for peer, corr in ranked[:8]:
                try:
                    p_industry = yf.Ticker(peer).info.get(
                        "industry", "unknown"
                    )
                except:
                    p_industry = "unknown"
                print(f"    {peer}: {corr:.3f}  ({p_industry})")
        else:
            print(f"  No fallback found for industry: '{industry}'")
            print(f"  Proceeding with dynamic results")


    top_peers = [peer for peer, corr in ranked[:5]]
    final_corrs = [corr for _, corr in ranked[:5]]
    final_avg_corr = np.mean(final_corrs) if final_corrs else 0

    print(f"\n{'─'*48}")
    print(f"  Final competitors selected: {top_peers}")
    print(f"  Final avg correlation:      {final_avg_corr:.3f}")
    print(f"  Fallback used:              {fallback_triggered}")
    print(f"{'─'*48}")


    comp_df = pd.DataFrame()
    for peer in top_peers:
        try:
            data = yf.download(
                peer, start=START_DATE,
                end=END_DATE, progress=False
            )
            comp_df[f"{peer}_return"] = data["Close"].pct_change()
            comp_df[f"{peer}_price"] = data["Close"]
        except:
            continue

    if not comp_df.empty:
        comp_df.index = pd.to_datetime(
            comp_df.index
        ).tz_localize(None)

    return comp_df, top_peers

def build_signal_matrix(ticker):
    print(f"\n{'='*55}")
    print(f"  Building signal matrix for {ticker}")
    print(f"  Period: {START_DATE} → {END_DATE}")
    print(f"{'='*55}")

    stock = yf.Ticker(ticker)
    company_name = stock.info.get("longName", ticker)
    print(f"  Company: {company_name}")


    price_df = get_price_data(ticker)
    macro_df = get_macro_data()
    sector_df, etf = get_sector_data(ticker)
    trends_df = get_trends_data(ticker, company_name)
    comp_df, peers = get_competitor_data(ticker)


    print(f"\n[6/6] Combining all signals...")
    combined = price_df.copy()

    def safe_join(left, right, label=""):
        if right is None or right.empty:
            return left
        dupes = [c for c in right.columns if c in left.columns]
        if dupes:
            print(f"  ⚠ Dropping duplicate columns from {label}: {dupes}")
            right = right.drop(columns=dupes)
        if right.empty:
            return left
        return left.join(right, how="left")

    combined = safe_join(combined, macro_df, "macro")
    combined = safe_join(combined, sector_df, "sector")
    combined = safe_join(combined, trends_df, "trends")
    combined = safe_join(combined, comp_df, "competitors")


    company_info = yf.Ticker(ticker).info
    industry_df = get_industry_signals(ticker, company_info)
    combined = safe_join(combined, industry_df, "industry")
    if not industry_df.empty:
        print(f"  Added {len(industry_df.columns)} industry-specific signals")


    combined = combined.ffill().bfill()


    combined = combined.dropna(subset=["return_1m"])

    print(f"\n{'='*55}")
    print(f"  Signal matrix complete")
    print(f"  Shape: {combined.shape[0]} rows × {combined.shape[1]} columns")
    print(f"  Date range: {combined.index[0].date()} → {combined.index[-1].date()}")
    print(f"  Signals: {list(combined.columns)}")
    print(f"{'='*55}")


    filename = f"{ticker}_signals.csv"
    combined.to_csv(filename)
    print(f"\n  Saved to {filename}")

    return combined, company_name


if __name__ == "__main__":
    ticker = input("Enter ticker: ").upper().strip()
    df, name = build_signal_matrix(ticker)
    print(f"\nPreview:")
    print(df.head())
