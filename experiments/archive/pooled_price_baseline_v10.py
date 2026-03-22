"""
Pooled Price-Only Baseline V10 - ICS_26 Leak Investigation.

Investigating suspected data leak from Ichimoku Chikou Span (ICS_26).

V10 FIX: Exclude ICS_26 from features.
- ICS_26 is the Chikou Span which plots current close 26 periods BACK.
- This could cause look-ahead bias if not handled correctly.

Builds on V9's temporal split fix.

Usage:
    python experiments/pooled_price_baseline_v10.py --epochs 100 --stocks 400
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pickle
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, matthews_corrcoef
from tqdm import tqdm
from collections import Counter

# V10: Features to exclude (suspected data leaks)
EXCLUDED_FEATURES = [
    'ichimoku_ICS_26',  # Chikou Span - plots close 26 periods back, possible look-ahead
    'dpo',              # Detrended Price Oscillator - suspected leak
]

try:
    from config import REPORTS_DIR, DATA_DIR, training_config, lstm_config
    from src.models import AttentionLSTM
    from src.training import Trainer
    from src.features.indicators_v7 import ComprehensiveIndicatorsV7
    RUNNING_LOCAL = True
except ImportError:
    RUNNING_LOCAL = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info(f"Python Version: {sys.version}")
logger.info(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    logger.info(f"CUDA Available: True, Device: {torch.cuda.get_device_name(0)}")

# Stock list (same as V8)
EXTENDED_TICKERS = [
"BKNG", "MELI", "FCNCA", "KLAC", "ASML", "MPWR", "COST", "ARGX", "EQIX", "REGN",
"META", "ESLT", "IDXX", "ULTA", "QQQ", "CASY", "DJCO", "SNDK", "MEDP", "ISRG",
"APP", "CVCO", "INTU", "MDGL", "CACC", "AXON", "VRTX", "UTHR", "SNPS", "NVMI",
"LIN", "WINA", "CRWD", "POWL", "TSLA", "MSFT", "CYBR", "MU", "STX", "DPZ",
"SMH", "LITE", "MDB", "IESC", "ROP", "SITM", "LPLA", "STRL", "SOXX", "TLN",
"ONC", "ALNY", "AMGN", "GOOG", "GOOGL", "SAIA", "AVGO", "IDCC", "AMAT", "LFUS",
"WTW", "WWD", "MAR", "PRAX", "VONE", "ADI", "VTHR", "HIFS", "CDNS", "ADBE",
"CME", "CEG", "ERIE", "KRYS", "AVAV", "FFIV", "FTAI", "NDSN", "AEIS", "LECO",
"EXPE", "WING", "RGLD", "RTH", "ITIC", "QQQM", "AAPL", "PODD", "PLPC", "OSIS",
"POOL", "ADSK", "UFPT", "MYRG", "WDC", "VTWG", "VRSN", "ADP", "TER", "AMD",
"AMZN", "LRCX", "MKSI", "QTEC", "ZBRA", "NTRA", "WDFC", "FSLR", "NXPI", "HON",
"ASND", "MTSI", "VSEC", "TTWO", "VRSK", "MUU", "TXN", "NXST", "DASH", "CHTR",
"BELFB", "EA", "ZS", "MORN", "JBHT", "PLXS", "BBH", "TMUS", "COIN", "PRN",
"CHRW", "APPF", "NVDA", "LGND", "MULL", "FIVE", "CTAS", "BELFA", "ROST", "MQQQ",
"AXSM", "SBAC", "CRAI", "WLFC", "PNRG", "ICLR", "STLD", "CHKP", "JKHY", "TXRH",
"BIIB", "PANW", "GLDI", "WDAY", "LOPE", "LULU", "IBB", "DHIL", "ODFL", "FCFS",
"VTWV", "ENSG", "MZTI", "IUSG", "MKTX", "HURN", "BCPC", "FAD", "SPYQ", "BLTE",
"GRID", "VICR", "JAZZ", "PSCI", "DAVE", "FUTU", "ALGN", "FTC", "FANG", "CINF",
"INSM", "ALAB", "PTC", "FTXL", "FSV", "BIDU", "IRTC", "NUTX", "QCOM", "MANH",
"COKE", "RGEN", "ESGU", "PEP", "NTRS", "LSTR", "ICUI", "DVY", "IJT", "PLTR",
"MIDD", "WTFC", "CAMT", "ACWI", "SANM", "MSTR", "ILMN", "SLAB", "PI", "PSMT",
"HLNE", "GILD", "QQEW", "BFC", "CIGI", "NBIX", "NOVT", "DUOL", "MASI", "PKW",
"TSEM", "HWKN", "SHOP", "TMDX", "PCTY", "BPOP", "FNX", "BTSGU", "QLYS", "CRDO",
"ABNB", "DXPE", "BOKF", "WLDN", "DDOG", "CRUS", "LAMR", "NTES", "UMBF", "CDW",
"PATK", "FRHC", "SUSL", "NVDU", "PYZ", "FEX", "CHCO", "PDP", "KBWP", "DORM",
"PCAR", "PLMR", "USLM", "KALU", "WRLD", "CLMB", "SKYY", "VONG", "TEAM", "FYX",
"IEI", "ENTG", "SIMO", "AEP", "WMT", "RMBS", "NXT", "OMAB", "DLTR", "QTUM",
"MNDY", "SENEA", "OLED", "NBN", "BNTX", "CAR", "EWBC", "NXTG", "EXE", "ABVX",
"GH", "GGLL", "SNEX", "TRI", "MGRC", "ROAD", "SHV", "SATS", "ASTS", "AIRR",
"OLLI", "BANF", "AIA", "PSCD", "PSL", "CELC", "WYNN", "NICE", "ESQ", "ARM",
"PPH", "TROW", "VTWO", "IUSV", "QQQE", "MIRM", "NUVL", "EXAS", "KTOS", "CPAG",
"UFPI", "RYTM", "TW", "ADUS", "PAYX", "PDD", "SAIC", "UAL", "NATH", "EXEEL",
"EEMA", "TCBI", "ESPO", "TTMI", "AAXJ", "HOOD", "PWRD", "ESGD", "DWAS", "CWST",
"QQXT", "FYC", "INCY", "KMB", "AUMI", "CHRD", "STRC", "ITRI", "TDIV", "GRAL",
"IRMD", "FELE", "WGS", "ORLY", "RVMD", "STRF", "SKYW", "CHDN", "NTAP", "EMB",
"IPAR", "AKAM", "NDAQ", "IEF", "FTCS", "VONV", "ACGL", "ROKU", "MBB", "USVM",
"CCB", "ULVM", "SATA", "VYMI", "FNY", "JJSF", "PFG", "MRCY", "AADR", "MSDD",
"CRWV", "SLVO", "FAB", "WFRD", "CFA", "AZN", "IPGP", "VIGI", "ONEQ", "ZM",
"AAON", "PSCM", "EBAY", "SBUX", "VC", "NVDL", "SMST", "CCEP", "DGRW", "ICFI",
"MCRI", "CASH", "QCRH", "FTA", "IXUS", "SPSC", "JTEK", "OTTR", "ACLS", "HAS",
"GPCR", "ALGT", "ARCB", "TLT", "FDT", "WIX", "SEIC", "FWONK", "JIVE", "CONI",
"NBIS", "LOGI", "HQY", "KBWB", "CVLT", "OKTA", "CARZ", "PTF", "AGYS", "RFDI",
"SDG", "STRA", "JSMD", "RFEM", "PFBC", "LMB", "PLUS", "LMAT", "IMCV", "VCIT",
"CROX", "BLLN", "BIB", "NFLX", "RRBI", "SHY", "CATH", "SIGI", "IONS", "CALM",
"NSIT", "LRGE", "CTSH", "PTGX", "SCZ", "SYNA", "SSNC", "LLYVK", "STRK", "LSCC",
"MMSI", "DOX", "RING", "VOTE", "ANIP", "INDB", "FTNT", "IOSP", "COO", "VCSH",
"MNST", "EMXC", "CSGS", "SMTC", "PTNQ", "FWONA", "VXUS", "JBSS", "STRT", "TTAN",
"LLYVA", "MRVL", "DRUG", "DMXF", "MGEE", "RKLB", "MLAB", "VPLS", "GEHC", "CSCO",
"VCRB", "VTC", "CMPR", "KSPI", "SAFT", "IRON", "QRVO", "BBIO", "SPRB", "ISHG",
"VCLT", "JSML", "EVRG", "MCHP", "VGUS", "VBIL", "CFO", "PSET", "JPEF", "BNDP",
"XEL", "DSGX", "PTCT", "ATRO", "HOLX", "IBKR", "DOO", "PVLA", "WOOD", "HSIC",
"SLMBP", "BND", "IMKTA", "AGMI", "CFFI", "DXCM", "ANPA", "TAYD", "STRD", "KYMR",
"RNA", "XT", "CDL", "EEFT", "PSIX", "RDVY", "IEUS", "REG", "PHO", "FJP",
"INBX", "HALO", "BKCH", "FOXA", "EXPO", "RBCAA", "PABU", "FORM", "STEP", "ACWX",
"SFM", "RDNT", "RYAAY", "VCTR", "URBN", "AVXC", "FRPT", "CDC", "NVEC", "CIBR",
"BNDW", "UIVM", "ARWR", "JGLO", "IBOC", "CRVL", "HWC", "GRVY", "CECO", "ACLX",
"ESTA", "QMOM", "FER", "VWOB", "USMC", "PABD", "NWE", "FGM", "TRMB", "KRUS",
"MCHPP", "EMCB", "SYBT", "NWPX", "AMUU", "FV", "LNTH", "SRCE", "SIXG", "SQQQ",
"KBWR", "WSHP", "AUGO", "VRTL", "LNT", "APGE", "SEZL", "BHRB", "FEUZ", "BITS",
"JOYY", "FOX", "LIVN", "WSFS", "FDTS", "THFF", "FPXI", "TARS", "AMBA", "BHF",
"IGF", "CYTK", "CZFS", "SLVR", "INTW", "RUSHA", "FISV", "TECH", "SOXQ", "HLAL",
"FLEX", "Z", "SLGL", "RRR", "GSHD", "TBBK", "MMYT", "LEGR", "LDEM", "MNPR",
"MCHI", "ZG", "AVT", "SMBC", "CHEF", "CTBI", "CTEC", "BANR", "GSAT", "ANDE",
"CSB", "ALNT", "CSGP", "GSBC", "SOLS", "TCOM", "AVUQ", "QQHG", "HQGO", "UTMD",
"ROOT", "TIGO", "CHPX", "AFRM", "TEM", "FYT", "NOVTU", "PSCT", "CHMG", "TRNS",
"FTSM", "VGIT", "PSC", "QABA", "HBCP", "ON", "UEVM", "JEPQ", "FTDS", "LKFN",
"IUS", "ZION", "PSCU", "USXF", "VGSH", "AMWD", "FTDR", "TPG", "EBI", "CG",
"DIOD", "CNXN", "IPKW", "PFI", "FNK", "MDLZ", "FBNC", "RAPT", "KLIC", "TSMX",
"FEP", "IBOT", "RUSHB", "FIGR", "FBIZ", "TREE", "CAKE", "VSMV", "CHPS", "EFSC",
"BOTT", "NYAX", "ESEA", "XMTR", "LIF", "NFTY", "BMRN", "SOCL", "INOD", "HUT",
"VGLT", "FROG", "ACMR", "KAT", "ASO", "YLDE", "VSDA", "QQUP", "DWUS", "FFUT",
"BKR", "ECPG", "GGAL", "SWKS", "SYM", "VEON", "ENLT", "QQQA", "AOTG", "DGCB",
"COLM", "DNTH", "SFST", "DFGP", "GSIB", "FKU", "QQQH", "FEIM", "IGIB", "FCAP",
"QQQI", "TQQQ", "BLKB", "UNTY", "PCVX", "QQA", "COCO", "GPIQ", "IREN", "DGRS",
"IGSB", "GPIX", "DFGX", "GTPE", "CGON", "DOCU", "AFJK", "CELH", "PFM", "MNZL",
"CBSH", "PYPL", "PY", "PIZ", "AIQ", "XPEL", "CLOA", "ROBT", "USIG", "ATLC",
"ZHOG", "SDSI", "MSEX", "TAXT", "TAXI", "BREM", "LAYS", "BRHY", "MBWM", "TAXE",
"PFIS", "EVSD", "PNQI", "FMB", "HTO", "BRNY", "GPRF", "ZTRE", "ZTEN", "UYLD",
"CRSP", "FMUB", "QVAL", "SMCZ", "BRTR", "NEGG", "USTB", "TSMU", "SLQD", "ZTWO",
"OPPJ", "ICOP", "CATY", "USSH", "CALI", "BALQ", "FMUN", "PCMM", "TMSF", "FLDB",
"WABC", "FRAF", "TSCO", "IPX", "HYBI", "OBIL", "USOI", "LMBS", "BLBD", "PMBS",
"ZMUN", "BCLO", "XBIL", "PXI", "EMXF", "TATT", "KARO", "TRBF", "TBIL", "PLBC",
"RBIL", "LMTL", "PSCE", "VTIP", "CRNX", "ACNB", "FCAL", "LGIH", "QCLN", "TCBK",
"FITB", "SKOR", "PLTU", "UFIV", "UCYB", "UNIY", "AMKR", "ISTB", "ALRM", "BNDX",
"OMCL", "UTWO", "SUPN", "PTH", "ASTE", "FMHI", "BGRN", "VNQI", "ISBA", "GNOM",
"MODL", "MEOH", "PRFZ", "EMEQ", "ZEUS", "HTHT", "ESGE", "LBRDK", "OZK", "HUBG",
"INTC", "UITB", "BWFG", "FDCF", "BABX", "LBRDA", "HYXF", "VMBS", "RKLX", "LGN",
"CAC", "ENZL", "INDY", "GBUG", "IBIT", "SNY", "IUSB", "PIO", "ANAB", "EGGQ",
"LASR", "POWI", "RDVT", "BL", "DAX", "FCVT", "QLDY", "COPJ", "COLL", "MOFG",
"MILN", "TARK", "AAOI", "FTSL", "VSAT", "ELVR", "UFO", "HERD", "CENX", "HISF",
"FEMS", "IQQQ", "ITRN", "SPBC", "SKWD", "JOUT", "SQLV", "WGMI", "GNMA", "CASS",
"PUI", "FPA", "TEKX", "SDTY", "GLPI", "MDLN", "MAZE", "FIXD", "EXC", "SRRK",
"BSVN", "NBTB", "BATRA", "MRNA", "GDS", "FEPI", "PSCH", "UTEN", "SHOO", "ACIW",
"XAIX", "BRKR", "PEGA", "KNSA", "VERA", "BLCR", "DEMZ", "FDBC", "UCTT", "UTWY",
"CTRN", "NZAC", "LMNX", "DGII", "QDTY", "FAST", "TRST", "PPC", "QQMG", "LQDA",
"SCSC", "BOEU", "APEI", "IGOV", "DDIV", "GFS", "EWJV", "STBA", "GKAT", "TRMK",
"UBSI", "PCH", "AGEM", "NATO", "SION", "AVGX", "VNOM", "LVHD", "GABC", "TDI",
"BUFC", "MBIN", "GVLE", "HYLS", "CCBG", "BJRI", "IMOM", "FMBH", "RPRX", "TUR",
"FSBW", "CXSE", "SPFI", "HOOG", "EXEL", "INDH", "HROW", "LINT", "OVBC", "UTHY",
"KBDU", "BUFI", "ALCO", "MOOD", "GUSE", "PRGS", "TWST", "XENE", "SPRX", "FDTX",
"DRS", "FICS", "GCT", "RDTY", "CPRT", "SDVY", "GLNG", "PAHC", "GSGO", "VABK",
"BATRK", "ACT", "MRX", "CSTL", "KEQU", "GTOP", "COPP", "NTNX", "FDIG", "AVL",
"CORT", "UPST", "BUFM", "SMHX", "PDEX", "FLXS", "FSBC", "SLNO", "BTSG", "CGNX",
"FSUN", "FRME", "ASMG", "VFLO", "AZTA", "CEFA", "WRND", "BEG", "AIFD", "TSBK",
"AMAL", "NMIH", "EXLS", "NCSM", "FTXO", "EFSI", "FVC", "QQQJ", "QBIG", "EUFN",
"ZD", "VALU", "HYDR", "QSIX", "SNSR", "HYMC", "VCYT", "WISE", "IMOS", "AIPI",
"ELFY", "FUNC", "CWCO", "AFOS", "XOMX", "TTEK", "KNGZ", "FBL", "BBSI", "TEKY",
"CNXC", "MBX", "CCNR", "NVDD", "BOTZ", "IBEX", "CSX", "NSSC", "OPTZ", "AMSF",
"METU", "NVAWW", "CART", "ENPH", "GLIBA", "AMZU", "SMID", "AGNG", "APOG", "LEXI",
"VCEL", "ALGM", "PEBK", "ATRC", "GLIBK", "HRMY", "ROE", "FSCS", "GLBE", "NKTR",
"NSI", "ORR", "BTGD", "NKSH", "RNRG", "IAC", "ATAT", "COGT", "BPRN", "OZEM",
"FRDU", "PECO", "FDIF", "FTGS", "GEME", "CANC", "DVLU", "AGIX", "CPLS", "UFCS",
"FNWD", "VTVT", "FIBK", "FMET", "DVOL", "LFSC", "LINE", "ORRF", "HRTS", "COWG",
"FCBC", "GTLB", "VLGEA", "BSY", "FMTM", "QQQS", "FBOT", "CBFV", "TOWN", "PLAB",
"IBCP", "INDV", "WSBC", "QOWZ", "FDFF", "PZZA", "SCHL", "WDGF", "BDGS", "BELT",
"URNJ", "TRS", "RIGL", "APLD", "SETM", "PLTZ", "SMCF", "AMID", "BSRR", "IDEF",
"WERN", "RUNN", "TERN", "SEEM", "PDBA", "ALKS", "HWBK", "BNR", "BILI", "LAUR",
"OPCH", "GLPG", "WASH", "KOID", "SEIE", "INTA", "FIZZ", "WTIP", "AXGN", "FTXH",
"DGRE", "DCOM", "ORKA", "BWMN", "JPY", "COWS", "IVAL", "SBCF", "WDAF", "MBCN",
"CENT", "CEPI", "FDNI", "ODD", "AMZZ", "TTEQ", "GBFH", "PSCC", "YORW", "ARTNA",
"FSFG", "KROP", "GINX", "CORO", "LDRX", "AROW", "FISI", "PDFS", "UVSP", "MPB",
"TXUE", "THRM", "LKQ", "RUSC", "VRDN", "WAFD", "FPXE", "IMCR", "MBUU", "IBAT",
"TURF", "INRO", "CARG", "OFLX", "QTOP", "KEAT", "GMAB", "VSNT", "MSFU", "THMZ",
"REMG", "WSML", "FCA", "ENDW", "PEBO", "IDYA", "DRIV", "VOLT", "TRUP", "AHMA",
"SYRE", "AMDG", "RCKY", "OVLY", "UMMA", "EBIZ", "ORCS", "SKYU", "PRDO", "LQDT",
"SKYT", "DMAT", "GLOW", "PFF", "AAVM", "FFIN", "TMET", "VECO", "MLYS", "TILE",
"TYRA", "AVGU", "SEDG", "NAMS", "FLNC", "TVTX", "WEYS", "PGC", "GQQQ", "FTXN",
"CSCL", "PBEU", "SARK", "UPB", "QTR", "NWS", "MTCH", "SPAM", "ICHR", "GRW",
"OBT", "CFLT", "RGLO", "VRNS", "RINT", "FEMB", "RVNL", "ZAP", "CENTA", "INTG",
"RDTL", "TTD", "STOK", "WCLD", "CBNK", "DALI", "AMSC", "GLSI", "KE", "SFLO",
"IFLO", "CLFD", "CANQ", "FSTR", "NWFL", "METL", "PGJ", "XCNY", "BETR", "WMG",
"HERO", "TGTX", "PATN", "ANGL", "EQPT", "QBUF", "TSAT", "DYTA", "YNOT", "FEM",
"NNE", "BULD", "STRS", "BBB", "SMCI", "CLOD", "PSTR", "TMED", "BCML", "FAAR",
"AAPU", "VKTX", "CMCSA", "GRIN", "GBLI", "JMID", "TSLR", "SEIS", "GIII", "FCCO",
"PBQQ", "ISUL", "ETOR", "COLB", "IBBQ", "TAX", "PRCT", "TERG", "HTFL", "COFS",
"PQOC", "CFBK", "QCLR", "TCMD", "TEXN", "ALLW", "EMIF", "IREG", "COHU", "BU",
"BUG", "PPIH", "SWP", "CBRL", "YOKE", "FFBC", "GLCR", "JD", "OCS", "FBRX",
"SMCO", "HEAL", "BEAM", "ISTR", "RAA", "PBOG", "MEMA", "BASV", "FTAG", "WILC",
"EZMO", "DSGR", "EWTX", "GFLW", "CHGX", "AAPB", "MVBF", "BFST", "QNXT", "FIVY",
"GOU", "CHSCP", "FDIV", "QQQX", "JHAI", "APPN", "VITL", "AVBH", "DUKX", "MCBS",
"QXQ", "NTCT", "DKNG", "GLXY", "MATE", "SPIT", "KQQQ", "TSMG", "FWRD", "DVSP",
"USAF", "WBD", "PHVS", "PPTA", "QQJG", "FALN", "ERET", "BEDY", "GOVI", "QYLG",
"CPB", "CCNE", "AGIO", "FHB", "GDFN", "MEMS", "PNTG", "UMBFO", "TSEL", "IROQ",
"LARK", "BAFE", "NEWZ", "QGRD", "COMT", "JFB", "CRWL", "KDP", "SNWV", "FTAIM",
"EXUS", "CAFG", "QDEL", "LINC", "SLM", "NVDS", "GDEN", "PKBK", "FCTE", "RAUS",
"FINX", "BNAI", "HBT", "NWSA", "GLDB", "REIT", "TRUD", "IFV", "BSJV", "FNLC",
"OBTC", "FMED", "IVSS", "BKMI", "RIFR", "HYP", "WTFCN", "SYZ", "DCOMG", "EGBN",
"CNOB", "BMRC", "RAPP", "CCSO", "WAY", "ODDS", "HAFC", "NPFI", "SPCT", "FMAO",
"ELVN", "AMPH", "WCBR", "RTXG", "BUSEP", "BANFP", "CHSCO", "ECOW", "EYE", "ATEX",
"TDSC", "CBK", "OTEX", "AEHR", "SMOM", "IMVT", "QQWZ", "PBRG", "FLGT", "FLN",
"AMUN", "VLYPN", "BSJU", "PBPH", "BDYN", "FXNC", "EASY", "ELIL", "IALT", "XOMAP",
"TRUT", "BSSX", "MATW", "ATLO", "PIE", "AGNCZ", "RGTZ", "GTR", "WSGE", "CCLDO",
"CHSCL", "EZRO", "APA", "DBX", "IBTP", "BSJW", "CAIQ", "MGIC", "OXLCI", "CDIG",
"HFWA", "TEXX", "BKMS", "TMB", "FITBI", "BSMZ", "WSBCO", "MBNKO", "ADAMH", "OLMA",
"FRWD", "USDX", "LOTI", "MSBIP", "GAINI", "QALT", "TSPY", "AGNCN", "HBANL", "FTAIN",
"MBINM", "VIASP", "BPOPM", "AGNCO", "WTBN", "NEWTI", "BSJX", "OXLCG", "ATLCZ", "TRINI",
"IBTQ", "NEWTG", "CHSCN", "METCZ", "TUGN", "PFOE", "ADAMM", "AGGA", "RWAYZ", "NCPB",
"MFICL", "ONBPP", "TRINZ", "SUSB", "ASMB", "XOMAO", "DVGR", "RWAYL", "ARQT", "BSMW",
"CHYM", "ACAD", "ALMS", "GECCI", "TPGXL", "XOMA", "OXSQH", "METCI", "VLYPP", "OCCIM",
"CVGW", "BDVL", "NEWTH", "ADAMG", "DRNZ", "MYCH", "NMFCZ", "PFDE", "ADAMI", "TMUSL",
"GLTO", "AGNCL", "MYCG", "MYCI", "FTGC", "VRIG", "MYCF", "CA", "VLYPO", "ZVZZT",
"CDNL", "IGIC", "HBDC", "AGNCP", "ADAMO", "MYFW", "CHSCM", "CRMT", "BCTK", "DMLP",
"HNNAZ", "OCCIO", "BMOP", "RILYK", "GAINN", "BSVO", "AGNCM", "ATLCL", "NATR", "MNSBP",
"UNB", "KMTS", "IMNM", "MYMI", "FITBP", "BASG", "LYEL", "IONL", "INBKZ", "XELLL",
"TWFG", "IBGA", "GIND", "BSMY", "GECCG", "IBGB", "PALD", "OXLCP", "OXLCN", "TCHI",
"CNOBP", "OCCIN", "RGC", "MBINL", "OXLCZ", "CLDX", "FLY", "AIPO", "GEGGL", "ZUMZ",
"TNXT", "CPRX", "TDSB", "PRMR", "FGNXP", "IBTO", "DUKH", "MGPI", "BUSE", "DTCR",
"VIAV", "JAPN", "ALRS", "AAPG", "ATLCP", "PRPO", "QQQG", "QCMD", "OXLCL", "IBGL",
"AVGG", "TRMD", "BTF", "ONB", "POWWP", "IFGL", "RDWR", "BLCN", "CNTA", "CAI",
"KRT", "HCOW", "GEN", "LBRDP", "CCOI", "ARLP", "NEMG", "CLSM", "HSAI", "BCFN",
"CWBC", "TSYX", "CBC", "IND", "NRIM", "NEWTP", "NFXL", "OXLCO", "BIDG", "PGNY",
"CNQQ", "BSMR", "FTHI", "FRPH", "RICK", "BSMQ", "PFXNZ", "OS", "ATNI", "CIVB",
"RGS", "IBGK", "BSMS", "NECB", "SFD", "RILYN", "SUSC", "FCEF", "FDRX", "SEPN",
"RARE", "CSWC", "REGCP", "BWAY", "TDOG", "OFSSH", "BRRR", "HIDE", "ADAML", "BSMT",
"KOD", "BSJQ", "KHC", "QQQY", "WTBA", "DYFI", "CVNX", "RKNG", "IBTM", "SSRM",
"BMAX", "SBLK", "IBTG", "TMUSI", "PRVA", "REYN", "TMUSZ", "IONZ", "GNTX", "ASTH",
"ALHC", "SOFI", "GOODN", "XMAG", "AVBP", "RILYG", "QURE", "NNNN", "HBANM", "IHYF",
"SOLC", "USAR", "GCBC", "PID", "WEEI", "ZYME", "BSJR", "HYZD", "AGZD", "DAKT",
"MSBI", "EVER", "IBTH", "CCD", "SPCX", "VAVX", "PWP", "LFMDP", "TCX", "CAPR",
"CCEC", "IBTI", "ADAMN", "BRKU", "SRET", "FEAT", "PCB", "USGG", "CZNC", "AIRT",
"IIIV", "APLS", "BWIN", "HSTM", "PDDL", "NUGY", "HNDL", "BSMU", "PKOH", "AOSL",
"AVNW", "ELE", "FCNCO", "LWAY", "FTXG", "GARY", "BLFS", "UBND", "DBVT", "BSJS",
"IBTJ", "TENB", "LYTS", "SBFG", "CMGG", "NCNO", "ROIV", "CLMT", "NCIQ", "KLAG",
"NBTX", "SMTI", "AVDL", "NBBK", "DNLI", "FID", "EBMT", "BSJT", "BRZE", "LBRX",
"CCSI", "INBK", "BSCX", "PEY", "CARE", "BSMV", "MCFT", "RGCO", "BANX", "MSFL",
"ERNZ", "MNSB", "EZPW", "JCAP", "ACGLO", "TBLD", "CEVA", "OUST", "ALKT", "QCMU",
"TCBIO", "YBST", "GREEL", "SRZN", "EEE", "SSS", "BSCY", "XXX", "FCNCP", "CARY",
"BSCW", "FTQI", "CMCO", "SEMY", "BSCZ", "SHPD", "HOVNP", "NTGR", "LEGH", "CLOU",
"EFSCP", "LOGO", "SOFX", "CRVS", "JACK", "ARQQ", "BSCS", "RBB", "GLAD", "MAT",
"IBTL", "UAE", "MFVL", "FULT", "SRPT", "CBLL", "PHAR", "CZR", "AIRTP", "EFAS",
"PCRX", "CDNA", "ETHA", "EBC", "NRC", "GLUE", "AFBI", "UPWK", "GOODO", "MFMO",
"RCMT", "SFNC", "SIRI", "SNDX", "PAGP", "TXG", "JMSB", "WW", "MBINN", "VHC",
"TECX", "UBRL", "WYFI", "LIVE", "AKTS", "IRDM", "ARCC", "YBTY", "OKLL", "FITBO",
"ZIONP", "DADS", "IBTK", "LANDO", "INVA", "OSBC", "VRM", "AQWA", "PPI", "BSCR",
"AMYY", "ZBIO", "URGN", "QAT", "YBMN", "DCBO", "MFIG", "CVBF", "WSC", "YB",
"NTRSO", "LANDP", "CDZIP", "JSM", "BSCQ", "MLKN", "NFXS", "BWBBP", "NHPAP", "PRE",
"SHPU", "NESR", "TBRG", "OSW", "BIOA", "VRRM", "TNDM", "ADAMZ", "XP", "PENG",
"PGY", "CSQ", "FDUS", "AXTI", "BWB", "CSIQ", "CRTO", "AZYY", "LDSF", "CUSD",
"FRD", "EKG", "BBYY", "METC", "BOTJ", "TBPH", "KBAB", "PAA", "XOMZ", "TSCM",
"DHCNL", "MRBK", "AMDL", "ANEL", "BVFL", "STAA", "BSCT", "ISSC", "TSLQ", "GILT",
"UPBD", "USCB", "FULTP", "RUN", "LVLU", "LUNR", "NHPBP", "HCSG", "XOVR", "FBYY",
"FONR", "VERX", "PRTC", "BZ", "JAKK", "SMPL", "XRPC", "NUG", "SBRA", "PRHIZ",
"MNRO", "SHBI", "SCVL", "OCFC", "HST", "MVLL", "APPX", "MDWD", "RGTI", "DGICA",
"HCKT", "FOXF", "TCBS", "SRAD", "DLLL", "ADEA", "ADPT", "SOHOB", "WSBF", "PICS",
"ICLN", "HNRG", "CORZZ", "KJD", "MOVE", "BCAL", "VISN", "GTX", "SHC", "PTIR",
"UUUG", "EML", "NGNE", "PLAY", "NVYY", "CZWI", "CORZ", "KROS", "ELIS", "DAPP",
"TSL", "MPLT", "TIPT", "YSPY", "ACGLN", "ESN", "QYLD", "DYN", "ACEP", "CNCG",
"BHFAL", "ACFN", "NIKL", "HBANP", "EVMT", "MGYR", "USAU", "SILC", "PLTG", "LE",
"NMRK", "QQQT", "ADMA", "FIVN", "HBNC", "NVDG", "TSLL", "HODU", "SNCY", "MESO",
"TNXP", "CGABL", "FINW", "DHCNI", "ECBK", "UPSG", "RTYY", "FTRI", "TWIN", "RYM",
"HBAN", "KIDS", "OCSAW", "RILYT", "PARK", "BHFAO", "SIGIP", "MFLX", "LEGN", "SLDE",
"SBU", "HDL", "RJET", "SLP", "LCNB", "BSCU", "LULG", "TRIN", "LYFT", "BOED",
"HURC", "BOEG", "DFTX", "SOHU", "ZLAB", "OZKAP", "BHFAP", "BCAX", "BSCV", "LIND",
"MXL", "LI", "ASYS", "WAFDP", "SRBK", "FRBA", "LMTS", "FTRE", "QCML", "NRIX",
"GDEV", "MKTW", "SIBN", "PDLB", "ILIT", "GEVG", "HELE", "LENZ", "ABNG", "PALU",
"TITN", "IONX", "PSNY", "CLBK", "LIFE", "GLDY", "SOHON", "SAIL", "PROV", "GHRS",
"MDIV", "STNE", "MLTX", "ACNT", "QRMI", "CIFR", "LTCC", "RILYZ", "BPYPM", "KBWY",
"HOOX", "ALMU", "INMD", "FTLF", "GEOS", "ORLG", "CPHC", "FWRG", "CLST", "ULH",
"PHOE", "KALV", "LSBK", "FFIC", "SATG", "QFIN", "TQQY", "BSET", "BPYPP", "DVAX",
"IMXI", "VBNK", "RIOT", "DRVN", "KMLI", "PESI", "LTBR", "MFI", "ULTI", "RGYY",
"JBIO", "PLYY", "VSOL", "OPXS", "FDSB", "RMR", "OESX", "FVCB", "MCHB", "ETON",
"KINS", "NPCE", "BRCB", "BRBI", "BPYPO", "AMRN", "AIP", "CCCXU", "MAMA", "HCM",
"DVAL", "SLRC", "MCRB", "OKTG", "CPZ", "EWZS", "CLBT", "AFYA", "EOSE", "GLDD",
"OMDA", "SMMT", "FUSB", "NTSK", "SIFY", "AVTX", "STRO", "CTNM", "REAL", "RIVN",
"ATEC", "VOD", "SAMG", "SGRY", "AUPH", "IOYY", "EXTR", "ALBG", "NN", "EDRY",
"GRRR", "ERII", "SUPX", "BBNX", "MGNI", "METD", "PMN", "PDBC", "CCAP", "SMX",
"PAX", "ASPC", "MTRX", "FOLD", "HYNE", "AMLX", "SBGI", "AEBI", "GRPN", "PCSC",
"LMNR", "PBFS", "SONO", "TFSL", "EYPT", "NIOG", "PTRN", "ESCA", "AAPD", "UONE",
"VTYX", "VREX", "SOGP", "MRAM", "KBWD", "BAND", "QBY", "PLSE", "UBCP", "PBHC",
"OPBK", "GAIN", "RMBI", "BYRN", "AMRX", "PONY", "GLRE", "BPYPN", "CCCX", "BHFAN",
"NODK", "SDGR", "HMYY", "SSBI", "JANX", "SPOK", "COEP", "DLO", "VOR", "FUTG",
"VELO", "PHAT", "HBR", "FRST", "CRML", "GBDC", "CRESY", "KRRO", "COTG", "AAL",
"FA", "QNST", "WNEB", "ARVN", "RELY", "KC", "AVO", "ACHC", "USGO", "TRIP",
"AALG", "RCAT", "OFIX", "ADUR", "ONEW", "MAXI", "HOFT", "FSEA", "DJT", "ORCU",
"LOVE", "CLPT", "BLFY", "WULF", "NTLA", "KRNT", "PMTS", "VTRS", "FMNB", "WVE",
"FBLA", "BTDR", "AEVA", "METCB", "LITP", "EH", "NEWT", "OPRA", "GLGG", "ANGI",
"ENTA", "NWBI", "SNAG", "MSFD", "LMRI", "HTBK", "DXR", "PRAA", "PENN", "PLUL",
"FLYW", "TENX", "CEPT", "LNSR", "BNTC", "VINP", "RDCM", "ADSE", "VLY", "AARD",
"MPAA", "NVCR", "CONX", "PLRZ", "XRAY", "MRTN", "TBMC", "XYZG", "LACG", "OCSL",
"PPHC", "CGO", "REFI", "ALTY", "CGBD", "NFBK", "ESHA", "MAAY", "MDRR", "OXLC",
"ORCX", "BCIC", "EVCM", "OOSB", "BBOT", "FBIOP", "XNCR", "HUTG", "BHFAM", "CARL",
"YQQQ", "CGEM", "RPD", "NRDS", "WBTN", "LNZA", "INKT", "HOPE", "NEO", "NVNO",
"RBKB", "DSP", "KVAC", "QBTZ", "CHA", "ASRT", "BAYA", "BKHAU", "TBCH", "SOLZ",
"YDES", "AVPT", "OSPN", "SMYY", "CHY", "TXXD", "MLACU", "ALCY", "UNHG", "OMER",
"CLSK", "SCZM", "PCYO", "OOQB", "RELL", "NAVN", "TNGX", "AGNC", "CORZW", "TXXS",
"WLAC", "CHACU", "SGA", "GOOD", "SHEN", "BODI", "SNBR", "TRDA", "OTLY", "VSTL",
"RDIB", "XPEG", "CSPI", "RGNX", "QPUX", "KYIV", "DAWN", "RAIL", "RAND", "OTGL",
"ALM", "BLBX", "CHI", "GCMG", "ANSC", "SPKL", "CRNC", "GPRE", "EURK", "MFIC",
"HQI", "LCID", "AOHY", "RTACU", "INSG", "ARRY", "PSKY", "TSMZ", "LBTYA", "LBTYK",
"PASG", "NXTC", "SMCX", "HOYY", "ACIC", "CHCI", "RILYL", "ATIIU", "IART", "SGML",
"SSYS", "BACCU", "SVAC", "ALDFU", "ERIC", "LAND", "DTSQ", "FRSH", "QVCGA", "CTLP",
"ZSTK", "SWBI", "SHIP", "FULC", "CAPN", "RTAC", "FACTU", "VCIC", "MACI", "PAMT",
"VACH", "UBFO", "TASK", "CCIX", "CGCTU", "DIME", "MBAV", "ATII", "BEAG", "BUUU",
"OACC", "CHAC", "CODA", "VGSR", "LIEN", "SIMA", "LPTH", "FTCI", "GIG", "COLA",
"PELI", "TRVI", "TSSI", "SERV", "BACQ", "KELYA", "TVA", "WB", "PLMK", "VNET",
"DRDB", "EMBC", "RANG", "FACT", "RIBB", "KG", "CEPO", "MLAC", "KFII", "DMAA",
"AGCC", "CEPF", "ELVA", "OPRX", "NHIC", "ONDS", "HSPT", "RAAQ", "FNWB", "ERAS",
"NOEM", "NPAC", "PMTR", "WENNU", "BLUW", "ETHM", "STRR", "IPEX", "ANGO", "CCXIU",
"GRAG", "EGAN", "MCGAU", "CGCT", "UYSC", "SZZL", "NTHI", "EVOXU", "HVII", "QSEA",
"ORIC", "MFIN", "HLXC", "LCCC", "CEPV", "HCACU", "XRPN", "IPOD", "BACC", "CAEP",
"DSGN", "TECTP", "DAAQ", "IEAGU", "PCAP", "ONCH", "CONL", "COIG", "HCMA", "AIRO",
"CRANU", "LAFAU", "TACO", "BPACU", "KBONU", "SLNHP", "BDCIU", "SCIIU", "BCAR", "PLBL",
"CRAQ", "SAAQU", "NEOG", "KOYNU", "MCGA", "WENN", "HTLD", "LOCO", "CHECU", "IGACU",
"KRAQU", "MBVI", "OYSE", "DRIO", "WSTNU", "DNMXU", "APXTU", "ARHS", "OIMAU", "FIGX",
"BLZR", "OTGA", "CBIO", "LWAC", "QUMS", "MMTXU", "DSACU", "SPRY", "BBCQU", "VNME",
"FGMC", "EVOX", "EMIS", "VHCPU", "DMIIU", "KBON", "MEVOU", "GIXXU", "SCPQU", "AEAQU",
"AHCO", "HSPTU", "MITK", "ITHAU", "KTWOU", "GSRF", "BLRKU", "MKLY", "GPACU", "MLAAU",
"PAL", "ADACU", "HNNA", "XRPI", "ALIS", "AFRI", "LPCVU", "PANG", "PTORU", "MUZEU",
"CRAC", "HCAC", "LAFA", "XCBEU", "ALOVU", "NBRGU", "RILYP", "SORNU", "RFIL", "IRHOU",
"GIW", "WSTN", "APXT", "MESH", "STRZ", "DYOR", "GPAC", "LFAC", "DSAC", "SGC",
"XSLLU", "VWAV", "BBCQ", "AEAQ", "RNAZ", "EXOZ", "TWLV", "BIXI", "HLIT", "NETG",
"AVS", "ENGN", "BDSX", "DCTH", "NAVI", "SNSE", "AMZD", "ALLT", "IMTX", "AVXX",
"OSS", "PCT", "SPCB", "SHLS", "JYNT", "ASUR", "DUOT", "HTCO", "ANTA", "CIFG",
"CYRX", "HLMN", "QUBT", "XBTY", "MARA", "SSSS", "INTR", "GT", "ADTN", "FUND",
"AEYE", "SUNS", "OCUL", "GO", "PSNL", "CCRN", "AMPL", "LAKE", "FSLY", "ANIK",
"DRH", "GRFS", "VALN", "RWAY", "QS", "KLRS", "KDK", "BGC", "FRMI", "NTIC",
"IMSR", "MBLY", "ALOT", "KPDD", "LPCN", "EXPI", "UONEK", "FDMT", "BIS", "SPOG",
"BVC", "CGNT", "CERT", "SHMD", "OSCG", "LZ", "NBIL", "AOUT", "MIND", "OSCX",
"NSYS", "SNCR", "ONDG", "PRTA", "INBS", "SPT", "SBET", "PYPG", "NVTS", "ESOA",
"GYRO", "SCOR", "ZVRA", "WLTH", "PERI", "MBS", "SNFCA", "INSE", "EKSO", "NVAX",
"LNKB", "ELTK", "SCLX", "RXST", "BOOM", "ANL", "TIGR", "RILY", "SOUN", "CRMG",
"NMFC", "SEVN", "NVA", "GWRS", "DDI", "CPSS", "LFCR", "TSLG", "AEC", "FGBI",
"RDACU", "MPG", "DERM", "BYFC", "PANL", "AUDC", "NVCT", "AVAH", "HBNB", "ETRL",
"CRBP", "FCEL", "AMDD", "UNIT", "PAVM", "KMDA", "ATCX", "ACET", "KURA", "GDYN",
"HIMX", "MLCI", "EPRX", "DMAC", "GYRE", "NOWL", "GTM", "TOYO", "SKRE", "GEMI",
"CHW", "ALGS", "TSDD", "RGTIW", "FORR", "ADAM", "WRD", "PRCH", "KYTX", "PLTD",
"CDRO", "WATT", "STHO", "BCBP", "ARDX", "BVS", "CRMD", "KRNY", "BRID", "GASS",
"GSIT", "RGTX", "IEP", "NCEW", "ELTX", "KYNB", "WKEY", "ADXN", "SOUX", "WEN",
"LILAK", "XBP", "MTEX", "AMCX", "VHUB", "RLAY", "VIR", "LILA", "ALAR", "CCC",
"MGTX", "QNRX", "STI", "JG", "FENC", "QUIK", "VMD", "BAIG", "VNDA", "TLRY",
"IMA", "OPEG", "ASLE", "ARCT", "RSVR", "RDAC", "PROF", "PTEN", "XXII", "CSBR",
"LXEO", "APVO", "PUBM", "XERS", "TLX", "DPRO", "SENS", "MNTS", "CFFN", "CVKD",
"SEAT", "DBGI", "SAIH", "TARA", "UG", "REPL", "VRCA", "HDSN", "MTYY", "TIL",
"LFST", "RENT", "ADBG", "CXDO", "BULL", "GECC", "SSTI", "DRTS", "VTIX", "AENT",
"BAFN", "ENVX", "MOMO", "AZ", "VIRC", "TH", "USOY", "RNAC", "TMC", "CVRX",
"KVHI", "NVD", "DUOG", "LTRX", "SIGA", "HRZN", "CMPS", "WHF", "JRVR", "IMMR",
"ASPI", "IMDX", "ECOR", "NUAI", "HELP", "EMAT", "MLEC", "BCRX", "KPLT", "NNAVW",
"ANNX", "KPTI", "EHLD", "ULBI", "CLRO", "NPT", "JFIN", "DOYU", "MRCC", "ZBAI",
"SLDB", "FWDI", "PDYN", "BCYC", "FGI", "PBYI", "QTRX", "PXLW", "MOB", "PAYO",
"CMPX", "IBRX", "NBIG", "MASS", "GGLS", "QMCO", "SWIM", "DWSH", "CRMLW", "XNET",
"PCLA", "SGHT", "NCTY", "SONM", "ICCC", "YDDL", "ODYS", "FBYD", "NEXN", "VSTM",
"IVA", "DJTWW", "KZR", "EVLV", "AVR", "NAGE", "MAAS", "BOLT", "MLCO", "RVYL",
"CCCXW", "SUGP", "DMRC", "IMMX", "DKNX", "YI", "BLMN", "DLHC", "UNCY", "CADL",
"STGW", "GLOO", "OPTX", "PLSM", "POET", "MNKD", "INGN", "BBCP", "CD", "FIP",
"SGMT", "TZOO", "NB", "DOMO", "ZURA", "PRTH", "ARBB", "FOFO", "RUM", "DHC",
"XWIN", "COYY", "BULX", "IDN", "SLE", "CNSP", "CING", "CTMX", "QTI", "MB",
"SLNG", "GBIO", "III", "OWLS", "CDZI", "PTLO", "MYGN", "MGRT", "FHTX", "TLS",
"INNV", "ALT", "CMTL", "PTON", "MTLS", "MCW", "FLYE", "AURA", "ATHR", "SVRA",
"ALDX", "KZIA", "RNW", "LEE", "ACOG", "ALVO", "AMCI", "KRMD", "OPRT", "CPSH",
"LFVN", "DIBS", "ARMG", "SOLT", "NEXT", "ASPS", "TTGT", "ILPT", "ARKO", "SOPH",
"CRVO", "TSLS", "OPEN", "WAVE", "APPS", "VOXR", "TCPC", "OCC", "ATLX", "RVSB",
"PEPG", "TDUP", "CRSR", "ONEG", "MDXG", "FIEE", "SRTS", "AIOT", "ABEO", "BBGI",
"OM", "ACDC", "POCI", "DFDV", "AVXL", "SELF", "IPWR", "ASTI", "BNC", "ELWT",
"SPAI", "CV", "ATRA", "TLSI", "CTKB", "TGL", "BOSC", "RDNW", "IRWD", "WHLR",
"ILAG", "EPSN", "RADX", "HTZ", "JBLU", "SRTA", "BHST", "AQMS", "NKLR", "LONA",
"IOTR", "BENF", "EMPD", "TSYY", "TPCS", "CVV", "WKHS", "SLN", "UDMY", "HERE",
"SFIX", "GSM", "MST", "COYA", "NXTT", "NYXH", "LDWY", "ULCC", "THRY", "SATL",
"ALTI", "WEST", "MRAL", "IMRX", "MLGO", "OMSE", "NTRB", "OFS", "ACTU", "UROY",
"GAMB", "VTSI", "SANG", "EOLS", "SND", "LSAK", "KFFB", "MARPS", "LVO", "OCCI",
"GWAV", "PYPD", "SLDP", "PDC", "RPID", "CLYM", "LSTA", "QNTM", "CLIK", "SANA",
"STKL", "TSHA", "BLZE", "RGP", "GOGO", "LOAN", "BMM", "MKZR", "SVCO", "FOXX",
"HPK", "NEPH", "KOSS", "PGEN", "GDC", "CAAS", "CRCT", "IONR", "UPC", "SMCL",
"SXTP", "THAR", "SSII", "GURE", "SABS", "ELSE", "GRAB", "ABTS", "RCEL", "SPHL",
"PPSI", "FLWS", "IPW", "RMCO", "BCTX", "CRWG", "DTST", "NEOV", "BTOG", "PSIG",
"ACHV", "AUR", "ETHZ", "QCLS", "SBC", "KRKR", "ISPO", "GITS", "EDAP", "TAOX",
"ASTL", "RXRX", "XCUR", "SOTK", "MBRX", "CHNR", "XRPT", "NWL", "MSGM", "JXG",
"RBNE", "AIRG", "IMSRW", "NERV", "CLNN", "API", "SNT", "MIGI", "AVIR", "LAES",
"FKWL", "INDI", "MQ", "PAYS", "ABUS", "APYX", "BNKK", "MOLN", "NOMA", "BIYA",
"PLCE", "SQFTP", "ACB", "FEED", "TROO", "NGEN", "TTRX", "PURR", "SDST", "FNKO",
"TWG", "STTK", "NEUP", "TBLA", "ARTV", "HIMZ", "ORGO", "BIRD", "NAMM", "NEXM",
"TALK", "NAII", "BLSG", "QTTB", "DOMH", "ABAT", "UEIC", "ACTG", "PRME", "KGEI",
"EWCZ", "ARBK", "VYGR", "DTIL", "TVRD", "NDRA", "CAMP", "YHGJ", "CLLS", "ATAI",
"TORO", "NHTC", "DTI", "GOAI", "SLS", "CURI", "CPIX", "MUD", "RRGB", "XFOR",
"CLAR", "SHIM", "LIXT", "IBG", "ABCL", "JLHL", "FLNT", "ISPR", "ELAB", "EVO",
"SYPR", "TNMG", "YQ", "LINK", "TACT", "PLTK", "XGN", "WLACW", "BGM", "NTWK",
"QVCGP", "RLMD", "NCMI", "JL", "ESGL", "QIPT", "OBIO", "KYIVW", "GRCE", "ELPW",
"VERI", "MDXH", "RPAY", "IZEA", "ARQ", "BMHL", "KSCP", "ACRS", "ZENA", "REAX",
"GROW", "LRMR", "IMPP", "FOSL", "VMAR", "RMNI", "SSP", "RAY", "ACON", "RCKT",
"HTOO", "STEX", "CGTL", "BTQ", "LWLG", "BJDX", "IDAI", "LCDL", "ATLN", "NBP",
"NUWE", "ESPR", "HYPD", "GRI", "UXIN", "WTF", "RR", "AGH", "BMBL", "INVE",
"MRVI", "RAVE", "INV", "GAIA", "VFF", "TKLF", "NIU", "PLUR", "PXS", "ASTC",
"JYD", "SNOA", "ORMP", "RZLT", "IMNN", "NA", "RKLZ", "VFS", "ASRV", "ACIU",
"AREC", "DSWL", "SPPL", "DFLI", "HSCS", "BAOS", "MSTX", "LCUT", "SCWO", "CREX",
"ATHE", "LFMD", "SDOT", "CLRB", "IHRT", "DCX", "MAMO", "LITM", "MDBH", "AIRJ",
"FLX", "AEI", "EVAX", "SINT", "PETS", "TTEC", "JVA", "HHS", "BLIV", "GRDX",
"LRHC", "FBIO", "GGR", "NAAS", "DAIO", "EU", "LTRN", "EBON", "AISP", "DNUT",
"JRSH", "CMRC", "PLUT", "NTRP", "WVVIP", "MKDW", "EVGO", "BLNE", "BLRX", "ANIX",
"LPSN", "AMPG", "ENVB", "AQST", "CMCT", "TYGO", "HPAI", "SIEB", "RYOJ", "SHMDW",
"BOF", "PSNYW", "LICN", "PMAX", "BEEP", "BWEN", "ATOM", "ABSI", "CRWS", "KTCC",
"BGIN", "PCSA", "BAER", "SKBL", "BNRG", "IQST", "HUBC", "VIVS", "BGL", "AIRS",
"AGEN", "AGPU", "EHTH", "LX", "SOUNW", "FGNX", "DWTX", "FUFU", "TRVG", "AXG",
"LCFY", "ICMB", "KLXE", "GALT", "SERA", "SBLX", "HEPS", "SY", "GNPX", "GRAN",
"WVVI", "VNCE", "TOI", "ADAG", "OSUR", "PSEC", "IMMP", "RSSS", "OKUR", "AAME",
"GOVX", "MTC", "SIDU", "NNOX", "TPST", "MMLP", "ANGH", "WHWK", "ANNA", "CCLD",
"BZUN", "GNLX", "TCRT", "XOS", "HSDT", "VSA", "HIVE", "WOOF", "AMBR", "FTEL",
"PODC", "CABA", "RBBN", "CNTX", "GEMG", "VUZI", "RZLV", "AYTU", "TONX", "ABOS",
"NCEL", "ENGNW", "MOBBW", "BRR", "BGLC", "IOVA", "MVST", "KOPN", "PULM", "AEMD",
"ARTW", "XHLD", "NCNA", "TKNO", "BULG", "ROMA", "LFS", "CRON", "XBIT", "MAXN",
"BDTX", "SKYX", "BULLW", "VERU", "HTLM", "MTVA", "NUKK", "HTZWW", "DGXX", "ALTO",
"HNST", "NIOBW", "RAIN", "UTSI", "DLTH", "HOLO", "VRA", "OPAL", "ZNTL", "WAI",
"BMRA", "BMNG", "CNCK", "FLL", "DRMA", "ACXP", "PFAI", "MENS", "CTRM", "VWAVW",
"CNTB", "SCAG", "IRD", "MSTP", "SQFT", "SNTG", "RMCF", "STKH", "BLDP", "CRNT",
"ICU", "ZGM", "BCG", "TMCI", "BITF", "GOSS", "CERS", "BTCS", "AFCG", "DRCT",
"KPRX", "DH", "HITI", "GDRX", "STKS", "SLGB", "TUSK", "XBIO", "CLOV", "PACB",
"SOHO", "NKTX", "SPRO", "NUAIW", "CHRS", "CYCU", "CRGO", "AZI", "CHSN", "INDP",
"WRAP", "ALZN", "SOPA", "ELDN", "CLNE", "TRAW", "CSTE", "LGCL", "PSHG", "EFOI",
"CMND", "FEAM", "TXMD", "GEG", "PLUG", "WIMI", "QRHC", "BRAG", "NRXP", "HCAT",
"FIGG", "JWEL", "STIM", "IBIO", "GOCO", "FORA", "ULY", "DTCX", "RKDA", "REVB",
"ALTS", "PROK", "GENK", "SORA", "FLD", "XRX", "BTMD", "FTFT", "IQ", "HUIZ",
"SHPH", "OKYO", "DWSN", "MATH", "VVPR", "SDA", "ABVC", "VGAS", "STSS", "CIGL",
"THCH", "UHG", "MTEN", "DXST", "MERC", "CELZ", "BMR", "AIMD", "CBUS", "EVTV",
"UBXG", "INAB", "GCTK", "SVC", "GDHG", "BTBT", "CNVS", "NMRA", "ABVE", "EDIT",
"DFSC", "LASE", "HERZ", "ICG", "ZJK", "NAUT", "OMEX", "SAVA", "IPSC", "JCTC",
"ABP", "GEVO", "WPRT", "FMST", "TAOP", "REBN", "VCICW", "AIFU", "HIND", "HKIT",
"GRNQ", "MIST", "LIQT", "DARE", "SAGT", "LYRA", "SNES", "SWAG", "CLGN", "SEGG",
"HOVR", "CCCC", "GNSS", "HFFG", "VBIX", "ASBP", "BTTC", "GANX", "CGEN", "IPHA",
"OXSQ", "SLXN", "ALEC", "WKSP", "EZGO", "AMST", "ECX", "OABI", "MBAI", "ARAI",
"KUST", "SEER", "AWRE", "ICON", "MGNX", "PRLD", "MBOT", "FATN", "RDZN", "ACRV",
"NNDM", "SWVL", "SNYR", "AIXC", "MYSE", "USEA", "LPRO", "NEON", "PROP", "AGMH",
"ALXO", "UPXI", "EPSM", "MNOV", "AUID", "PHUN", "SUUN", "MDAI", "NVNI", "BDRX",
"IVVD", "MGN", "TLSIW", "YMT", "IPM", "CYN", "WOK", "CHAI", "RDGT", "CJMB",
"WBUY", "HOUR", "ALLO", "SPWR", "CISS", "WAFU", "PIII", "CTW", "SUIG", "MNTK",
"VSME", "LIDR", "APWC", "WETH", "LGHL", "BON", "OCG", "XTIA", "GLXG", "CRDF",
"KMRK", "LUNG", "HYFT", "RECT", "FARM", "DLPN", "MCHX", "BBLG", "LEDS", "GLBS",
"EGHT", "POWW", "ZKIN", "POLA", "MRKR", "INO", "ARTL", "WALD", "TWAV", "BEEM",
"PRQR", "KLTR", "NSPR", "FAMI", "HWH", "STKE", "UCL", "FUSE", "CETX", "LHAI",
"MTEK", "BRLT", "VVOS", "RVMDW", "CREV", "YTRA", "KXIN", "RUMBW", "CMMB", "YJ",
"MGX", "BTAI", "BYSI", "RPGL", "GENVR", "SVACW", "CVGI", "INMB", "NVVE", "IKT",
"SNDL", "GIBO", "VEEE", "PNBK", "CRCG", "OVID", "NEOVW", "CDIO", "OLPX", "OGI",
"STAI", "MRM", "PLBY", "WNW", "CNTY", "FATBB", "GIGM", "PALI", "HYFM", "IRIX",
"KDKRW", "NNBR", "CMBM", "AMTX", "ABTC", "JSPR", "MGIH", "SKIN", "TLSA", "ORBS",
"XELB", "UCAR", "SEV", "GNLN", "PYXS", "CURR", "BEAT", "OCGN", "NWTG", "QETAR",
"USGOW", "RAYA", "SJ", "GXAI", "HAO", "EXFY", "LGCB", "NXGL", "DUO", "BZAI",
"AKBA", "AHG", "VANI", "CRBU", "USIO", "CNDT", "SATLW", "ENTX", "LAB", "EDUC",
"NIVF", "LESL", "LRE", "AUTL", "CYCN", "HGBL", "CHR", "YOUL", "EQ", "ONCO",
"VERO", "NWGL", "BTCT", "SPWH", "SLMT", "IZM", "AKAN", "TRON", "UPLD", "GERN",
"ACCL", "FLUX", "HRTX", "FTEK", "GRWG", "JDZG", "BTBD", "LZMH", "BNBX", "VIOT",
"YAAS", "BNGO", "EUDA", "PAVS", "NFE", "JZXN", "UOKA", "FRGT", "GREE", "IMCC",
"ETHMW", "ORIO", "RCON", "ORKT", "SLSN", "IVF", "CLWT", "CRESW", "NITO", "TMCWW",
"ORIS", "MRNO", "GV", "SABR", "NCI", "SNGX", "CELU", "YYAI", "RNTX", "MIRA",
"OPK", "DFDVW", "LNKS", "MEGL", "JCSE", "SEED", "LOT", "RANI", "SOBR", "SCKT",
"GVH", "CDXS", "BOXL", "VS", "RDHL", "OLOX", "LGO", "ARBE", "LOOP", "BCDA",
"INHD", "XAIR", "BGMS", "MDCX", "SBFM", "MNY", "AIHS", "ERNA", "SSKN", "CNET",
"REFR", "ELOG", "TRSG", "FNGR", "MYNZ", "PETZ", "LITS", "VRME", "GCL", "PLRX",
"BNZI", "BIVI", "CWD", "MNDO", "APRE", "TIVC", "HCWB", "TBHC", "HAIN", "OMH",
"GMM", "ANTX", "UFG", "HUDI", "EM", "KBSX", "GTIM", "BRTX", "GPRO", "BCTXL",
"TELO", "BOLD", "PRZO", "SHFS", "FATE", "WLDSW", "HIT", "MNDR", "BIAF", "IPDN",
"LUCD", "DTSS", "LXRX", "BMEA", "QSI", "SURG", "HYPR", "LUCY", "FTHM", "GSUN",
"MSW", "EDSA", "AACG", "SLNH", "CGC", "DOGZ", "INM", "YSXT", "CGTX", "PMVP",
"GIFT", "RCT", "TJGC", "REKR", "CREG", "CDT", "RDI", "IPST", "RZLVW", "CTOR",
"FEBO", "EDTK", "KITT", "RMTI", "DGNX", "RYET", "AISPW", "BDMD", "ESLA", "INTZ",
"XCH", "CTNT", "TOP", "LVRO", "AMOD", "OXBR", "ANEB", "CLPS", "USEG", "BTM",
"ALLR", "TCRX", "RETO", "ELUT", "FFAI", "DEVS", "FRSX", "KNDI", "GNTA", "OPTXW",
"ENLV", "ADV", "EVGN", "PHIO", "NIPG", "ZBAO", "SNTI", "ADGM", "SCNI", "SKYE",
"XHG", "HUMA", "CASI", "ADSEW", "NVX", "TELA", "COCP", "ONCY", "GELS", "HMR",
"HCHL", "BFRI", "RNXT", "INVZ", "BRLS", "SVRN", "MBIO", "RITR", "GAUZ", "SUNE",
"TLPH", "ZYBT", "GP", "LSH", "AQB", "SFWL", "AUUD", "CRDL", "MSGY", "PMEC",
"QH", "NXXT", "CALC", "CDLX", "ZEO", "VRAR", "HIHO", "GMHS", "WLDS", "CNEY",
"NISN", "INCR", "ALPS", "ENGS", "IMTE", "SMXT", "GTEC", "ELBM", "JF", "RMSG",
"MXCT", "DYAI", "PRSO", "DRTSW", "DLXY", "YIBO", "NRSN", "BIOX", "HLP", "AIFF",
"YDESW", "TNON", "TVACW", "IFRX", "ZNB", "SVRE", "APM", "IMRN", "NMTC", "YHC",
"TANH", "HOTH", "CRIS", "BZFD", "YDKG", "XTLB", "KTTA", "RIME", "ZENV", "PCTTW",
"IVDA", "FMSTW", "EJH", "BCARW", "AEHL", "CHACR", "MBAVW", "IMG", "TURB", "MWYN",
"AFRIW", "RTACW", "CGCTW", "PDSB", "FBGL", "OPENW", "ATYR", "CCIIW", "EPOW", "KWM",
"MVIS", "LBGJ", "CABR", "BHAT", "CCG", "CCTG", "BLIN", "ADTX", "TRIB", "GDTC",
"LNAI", "CSAI", "XRPNW", "AVX", "HKPD", "SGRP", "PMCB", "CBAT", "ARAY", "TNYA",
"DEFT", "IINN", "MAPS", "ONMD", "TOMZ", "DCOY", "XPON", "BYND", "MYSZ", "DCGO",
"ASST", "ALBT", "INEO", "AIRJW", "DRDBW", "JZ", "CCIXW", "MDCXW", "BACQR", "CYPH",
"BMGL", "CTXR", "CETY", "PELIR", "FTRK", "SPRC", "FCUV", "GMGI", "THH", "LEXX",
"RUBI", "ABVEW", "IFBD", "BKYI", "BLNK", "SCYX", "NCPL", "SNAL", "BRNS", "ATER",
"NIXX", "SMTK", "ABLV", "SFHG", "ZTEK", "CDROW", "PRHI", "GLMD", "GIPR", "DXLG",
"ENSC", "KBONW", "PRPL", "HOWL", "PRFX", "COOT", "DVLT", "NXPL", "JAGX", "TOUR",
"NDLS", "PBM", "ONFO", "GTBP", "TEAD", "CAN", "UK", "YXT", "ATON", "CTSO",
"IPEXR", "MDIA", "SOWG", "COCH", "SLDPW", "WETO", "OSRH", "RLYB", "JBDI", "TRUG",
"BFRG", "WTO", "CIIT", "RXT", "ICCM", "IMUX", "PPBT", "CLIR", "LOBO", "REE",
"MYPS", "LHSW", "LFWD", "AENTW", "CAPS", "MCGAW", "LIMN", "VEEA", "TGHL", "FGMCR",
"XLO", "SGLY", "PN", "NAMI", "VYNE", "LGVN", "VTGN", "CAPT", "NXL", "EVOXW",
"ATOS", "SMSI", "GFAI", "FEMY", "XRTX", "HURA", "MDAIW", "NCRA", "JUNS", "OLB",
"JFBR", "BAERW", "KALA", "PRTS", "WXM", "ETS", "BLUWW", "AERT", "VCIG", "TBMCR",
"FACTW", "HBIO", "ZDAI", "BTOC", "EDBL", "RAAQW", "BLZRW", "SGMO", "NOTV", "ITOC",
"GUTS", "CCHH", "HXHX", "ZOOZ", "TLNCW", "FATBP", "TRNR", "SCNX", "MODD", "COSM",
"MREO", "OPENL", "YMAT", "ASNS", "FERAR", "MBVIW", "WYHG", "CISO", "TBH", "DAAQW",
"OTLK", "NTCL", "CUPR", "ALVOW", "ZSPC", "COLAR", "WFF", "ZCMD", "GSHRW", "TLIH",
"WENNW", "AGRZ", "RANGR", "AMIX", "DAIC", "TACHW", "STAK", "OPENZ", "CPOP", "GAME",
"AFJKR", "INTJ", "LMFA", "AIRE", "HOVRW", "CLSKW", "MGRX", "GRRRW", "INLF", "BZAIW",
"INTS", "NAKA", "VSEE", "WGRX", "BEAGR", "BANL", "MESHW", "ISPC", "SILO", "ATXG",
"TACOW", "LATAW", "CURX", "CUE", "DKI", "CDTG", "FMFC", "DNMXW", "CRE", "BIVIW",
"FBLG", "HCAI", "TDTH", "OFAL", "AACBR", "PGYWW", "UHGWW", "BACCR", "CXAI", "IOBT",
"GLE", "RVSN", "ITRM", "IXHL", "BGLWW", "ANSCW", "AUROW", "ASPSZ", "LFACW", "SKK",
"KOYNW", "SKYQ", "XWEL", "MIMI", "AGAE", "VSTD", "ADVB", "SPKLW", "MSAI", "SPEGR",
"AURE", "RVPH", "TVGN", "BCAB", "HTCR", "SCIIR", "KCHVR", "NAMMW", "VRAX", "HCTI",
"FAT", "AREB", "ASPSW", "DAVEW", "FCHL", "DSY", "BNAIW", "MSS", "VNMEW", "AEAQW",
"GPATW", "KLTO", "ANY", "MASK", "GRABW", "XTKG", "DTCK", "PPCB", "EDHL", "GPACW",
"HCACR", "AKTX", "LPAAW", "ITHAW", "FUFUW", "EEIQ", "PSTV", "APADR", "SZZLR", "CRACR",
"POM", "NMPAR", "EHGO", "PASW", "LXEH", "BPACR", "YYGH", "EDBLW", "ADIL", "ANNAW",
"ESHAR", "AIIO", "NCT", "ODVWZ", "RMCOW", "MEHA", "ASPCR", "DTSQR", "MOBX", "EZRA",
"TDIC", "WSTNR", "TWNP", "AIXI", "WORX", "SAFX", "MTEKW", "KFIIR", "ORGN", "BETRW",
"QNCX", "RAINW", "CUBWW", "DTSTW", "BFRGW", "NOEMW", "KIDZ", "APACR", "EVLVW", "MACIW",
"ARQQW", "OYSER", "TDWDR", "JEM", "VFSWW", "GSIW", "QSIAW", "JTAI", "FGL", "UGRO",
"CENN", "BANXR", "ASTLW", "BCTXZ", "WCT", "CRACW", "ARBEW", "DMAAR", "ZJYL", "FOXXW",
"RENX", "FLDDW", "AMPGW", "PDYNW", "NUKKW", "GFAIW", "ONMDW", "BTBDW", "NXGLW", "SRTAW",
"HSCSW", "NXPLW", "STFS", "PTLE", "TIRX", "AEVAW", "APLT", "TOIIW", "KITTW", "EUDAW",
"KLTOW", "PFSA", "BYAH", "MVSTW", "CINGW", "HOLOW", "RVSNW", "MRNOW", "OXBRW", "LIDRW",
"AMODW", "VEEAW", "ATPC", "LOTWW", "KWMWW", "LUCYW", "RFAIR", "SMXWW", "NRXPW", "DFLIW",
"ONFOW", "XBPEW", "EVGOW", "MNYWW", "SXTPW", "BTMWW", "OUSTZ", "OSRHW", "SCAGW", "NVNIW",
"DAICW", "CAPTW", "MSAIW", "BNCWW", "SEATW", "VGASW", "ZEOWW", "VSEEW", "RMSGW", "AIIOW",
"GCLWW", "TVGNW", "COCHW", "TBLAW", "NCPLW", "WALDW", "GIBOW", "SLXNW", "SXTC", "ACONW",
"CXAIW", "FFAIW", "JSPRW", "ASBPW", "ABLVW", "STSSW", "DFSCW", "ICUCW", "TALKW", "DSYWW",
"PRENW", "MNTSW", "SQFTW", "JFBRW", "DRMAW", "COOTW", "HUBCW", "PBMWW", "NIVFW", "KIDZW",
"RCKTW", "BNZIW", "AREBW", "SVREW", "CDIOW", "NVVEW", "ABPWW", "REVBW", "BCTXW", "CREVW",
"KTTAW", "BENFW", "CELUW", "LTRYW", "MLECW", "TNONW", "SWVLW", "MAPSW", "ISPOW", "CDTTW",
"INVZW", "RNWWW", "XOSWW", "KPLTW", "ORGNW", "EZRAW"
]

EXTENDED_TICKERS = list(dict.fromkeys(EXTENDED_TICKERS))


class PriceCache:
    """Cache for yfinance price data."""
    
    def __init__(self, cache_dir: Path = None):
        if RUNNING_LOCAL:
            self.cache_dir = cache_dir or DATA_DIR / "price_cache"
        else:
            self.cache_dir = cache_dir or Path("/content/data/price_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, start_date: str, end_date: str) -> Path:
        return self.cache_dir / f"prices_{start_date}_{end_date}.pkl"
    
    def load(self, start_date: str, end_date: str, requested_stocks: List[str]) -> Dict[str, pd.DataFrame]:
        cache_path = self.get_cache_path(start_date, end_date)
        if cache_path.exists():
            logger.info(f"Loading cached prices from {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            logger.info(f"Cache contains {len(cached)} stocks")
            return cached
        return None
    
    def save(self, data: Dict[str, pd.DataFrame], start_date: str, end_date: str):
        cache_path = self.get_cache_path(start_date, end_date)
        if not cache_path.exists():
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)


class PooledExperimentV10:
    """
    V9: Proper temporal split - split EACH STOCK by time, THEN pool.
    
    Key fix: Every stock is split into train/val/test by DATE, not index.
    Train: 2014-2021 from ALL stocks
    Val:   2021-2023 from ALL stocks  
    Test:  2023-2024 from ALL stocks
    """
    
    def __init__(
        self,
        stocks: List[str] = None,
        start_date: str = "2014-01-01",
        end_date: str = "2024-12-31",
        train_end_date: str = "2022-01-01",  # Train: everything before this
        val_end_date: str = "2023-06-01",     # Val: train_end to this
        # Test: everything after val_end
        sequence_length: int = 20,
        epochs: int = 100,
        batch_size: int = 1024,
        early_stopping_patience: int = 15,
        dropout: float = 0.4,
        learning_rate: float = 5e-4,
        use_cache: bool = True,
        colab_mode: bool = False,
    ):
        self.stocks = stocks or EXTENDED_TICKERS[:400]
        self.start_date = start_date
        self.end_date = end_date
        self.train_end_date = pd.Timestamp(train_end_date)
        self.val_end_date = pd.Timestamp(val_end_date)
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.use_cache = use_cache
        self.colab_mode = colab_mode
        
        if colab_mode:
            self.output_dir = Path(f"/content/reports/pooled_v10_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            self.output_dir = REPORTS_DIR / f"pooled_v10_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.price_cache = PriceCache()
        
    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """Load stocks from cache or download."""
        stock_data = {}
        
        if self.use_cache:
            cached = self.price_cache.load(self.start_date, self.end_date, self.stocks)
            if cached:
                for ticker, df in cached.items():
                    if ticker in self.stocks:
                        stock_data[ticker] = df
                logger.info(f"Loaded {len(stock_data)} stocks from cache")
        
        missing = [t for t in self.stocks if t not in stock_data]
        logger.info(f"Need to download {len(missing)} additional stocks")
        
        if missing:
            batch_size = 100
            for i in range(0, len(missing), batch_size):
                batch = missing[i:i + batch_size]
                try:
                    data = yf.download(batch, start=self.start_date, end=self.end_date,
                                       group_by="ticker", threads=True, progress=True)
                    for ticker in batch:
                        try:
                            df = data[ticker].copy() if len(batch) > 1 else data.copy()
                            df = df.dropna()
                            if len(df) < 100:
                                continue
                            df.columns = [c.capitalize() if isinstance(c, str) else c for c in df.columns]
                            df['Ticker'] = ticker
                            stock_data[ticker] = df
                        except:
                            pass
                except Exception as e:
                    logger.warning(f"Batch download failed: {e}")
            
            if self.use_cache and len(stock_data) > 0:
                self.price_cache.save(stock_data, self.start_date, self.end_date)
        
        logger.info(f"Total: {len(stock_data)} stocks ready")
        return stock_data
    
    def prepare_stock_with_temporal_split(self, df: pd.DataFrame, target_feature_cols: List[str] = None) -> Tuple[
        np.ndarray, np.ndarray,  # X_train, y_train
        np.ndarray, np.ndarray,  # X_val, y_val
        np.ndarray, np.ndarray,  # X_test, y_test
        List[str]                # feature_cols
    ]:
        """
        THE KEY FIX: Split THIS stock by DATE, then create sequences.
        Returns train/val/test already split for this stock.
        """
        df = self.indicator_computer.compute_all(df)
        
        if target_feature_cols:
            feature_cols = target_feature_cols
            # Ensure all required columns exist, fill with 0 if missing
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
        else:
            feature_cols = self.indicator_computer.get_indicator_columns(df)
            
            # V10: Remove suspected leak features
            feature_cols = [col for col in feature_cols if col not in EXCLUDED_FEATURES]
            logger.debug(f"Excluded features: {EXCLUDED_FEATURES}")
            
            # Filter valid columns
            valid_cols = [col for col in feature_cols 
                          if col in df.columns and df[col].notna().sum() > 10]
            feature_cols = valid_cols
        
        if len(feature_cols) < 5:
            return (np.array([]), np.array([]), np.array([]), 
                    np.array([]), np.array([]), np.array([]), [])
        
        # Create target
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['trend'] = pd.cut(
            df['return_next'],
            bins=[-np.inf, -0.005, 0.005, np.inf],
            labels=[0, 1, 2]
        ).astype(float)
        
        df = df.dropna(subset=['trend'] + feature_cols[:10])
        if len(df) < self.sequence_length + 10:
            return (np.array([]), np.array([]), np.array([]), 
                    np.array([]), np.array([]), np.array([]), [])
        
        # Get feature data
        feature_data = df[feature_cols].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        labels = df['trend'].values
        
        # Get the index (dates) for temporal splitting
        dates = df.index
        
        # Create sequences with date tracking
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []
        
        for i in range(len(feature_data) - self.sequence_length):
            seq_end_date = dates[i + self.sequence_length - 1]  # Last day of sequence
            target_date = dates[i + self.sequence_length]       # Target day
            
            seq = feature_data[i:i + self.sequence_length]
            label = labels[i + self.sequence_length - 1]  # Fix: predict T+1, not T+2
            
            # Split by TARGET date (the day we're predicting)
            if target_date < self.train_end_date:
                X_train.append(seq)
                y_train.append(label)
            elif target_date < self.val_end_date:
                X_val.append(seq)
                y_val.append(label)
            else:
                X_test.append(seq)
                y_test.append(label)
        
        return (
            np.array(X_train, dtype=np.float32) if X_train else np.array([]),
            np.array(y_train, dtype=np.int64) if y_train else np.array([]),
            np.array(X_val, dtype=np.float32) if X_val else np.array([]),
            np.array(y_val, dtype=np.int64) if y_val else np.array([]),
            np.array(X_test, dtype=np.float32) if X_test else np.array([]),
            np.array(y_test, dtype=np.int64) if y_test else np.array([]),
            feature_cols
        )
    
    def prepare_temporal_pooled_data(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple[
        np.ndarray, np.ndarray,  # X_train, y_train
        np.ndarray, np.ndarray,  # X_val, y_val
        np.ndarray, np.ndarray,  # X_test, y_test
        List[str]                # feature_cols
    ]:
        """
        Process all stocks, split each by date, then pool train/val/test separately.
        """
        import gc
        
        all_train_X, all_train_y = [], []
        all_val_X, all_val_y = [], []
        all_test_X, all_test_y = [], []
        feature_cols = None
        success_count = 0
        
        for ticker, df in tqdm(stock_data.items(), desc="Processing stocks with temporal split"):
            try:
                # Pass feature_cols if established, otherwise let the function determine them
                X_tr, y_tr, X_val, y_val, X_te, y_te, cols = self.prepare_stock_with_temporal_split(df, target_feature_cols=feature_cols)
                
                if len(X_tr) == 0 and len(X_val) == 0 and len(X_te) == 0:
                    continue
                    
                if len(X_tr) > 0:
                    all_train_X.append(X_tr)
                    all_train_y.append(y_tr)
                if len(X_val) > 0:
                    all_val_X.append(X_val)
                    all_val_y.append(y_val)
                if len(X_te) > 0:
                    all_test_X.append(X_te)
                    all_test_y.append(y_te)
                    
                success_count += 1
                
                # Establish the feature columns from the first successful stock
                if feature_cols is None:
                    feature_cols = cols
                    logger.info(f"Established {len(feature_cols)} feature columns from {ticker}")
                    
            except Exception as e:
                if success_count < 3:
                    logger.error(f"Failed to process {ticker}: {e}")
        
        logger.info(f"Successfully processed: {success_count}/{len(stock_data)} stocks")
        
        if not all_train_X:
            raise ValueError("No training data available")
        
        X_train = np.concatenate(all_train_X, axis=0)
        y_train = np.concatenate(all_train_y, axis=0)
        X_val = np.concatenate(all_val_X, axis=0) if all_val_X else np.array([])
        y_val = np.concatenate(all_val_y, axis=0) if all_val_y else np.array([])
        X_test = np.concatenate(all_test_X, axis=0) if all_test_X else np.array([])
        y_test = np.concatenate(all_test_y, axis=0) if all_test_y else np.array([])
        
        del all_train_X, all_train_y, all_val_X, all_val_y, all_test_X, all_test_y
        gc.collect()
        
        logger.info(f"Temporal split results:")
        logger.info(f"  Train: {len(X_train):,} samples (before {self.train_end_date.date()})")
        logger.info(f"  Val:   {len(X_val):,} samples ({self.train_end_date.date()} to {self.val_end_date.date()})")
        logger.info(f"  Test:  {len(X_test):,} samples (after {self.val_end_date.date()})")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    
    def scale_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
        """Scale data - fit on train only."""
        logger.info("Scaling data (fit on train only)...")
        
        n_train, seq_len, n_features = X_train.shape
        
        X_train_flat = X_train.reshape(-1, n_features)
        X_val_flat = X_val.reshape(-1, n_features) if len(X_val) > 0 else None
        X_test_flat = X_test.reshape(-1, n_features) if len(X_test) > 0 else None
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
        
        if X_val_flat is not None:
            X_val_scaled = scaler.transform(X_val_flat)
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
        else:
            X_val_scaled = X_val
            
        if X_test_flat is not None:
            X_test_scaled = scaler.transform(X_test_flat)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
        else:
            X_test_scaled = X_test
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def predict_batched(self, model: nn.Module, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        model.eval()
        device = next(model.parameters()).device
        all_preds = []
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            with torch.no_grad():
                batch_tensor = torch.FloatTensor(batch).to(device)
                predictions = model.predict(batch_tensor)
                all_preds.append(predictions["trend_class"].cpu().numpy())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return np.concatenate(all_preds)
    
    def compute_metrics(self, y_test: np.ndarray, pred_classes: np.ndarray) -> Dict:
        class_counts = Counter(y_test)
        most_common_count = class_counts.most_common(1)[0][1]
        zero_rule_accuracy = most_common_count / len(y_test)
        
        accuracy = (pred_classes == y_test).mean()
        f1_macro = f1_score(y_test, pred_classes, average='macro')
        f1_weighted = f1_score(y_test, pred_classes, average='weighted')
        mcc = matthews_corrcoef(y_test, pred_classes)
        
        class_acc = {}
        for cls in [0, 1, 2]:
            mask = y_test == cls
            if mask.sum() > 0:
                class_acc[f"class_{cls}_acc"] = float((pred_classes[mask] == cls).mean())
        
        return {
            "accuracy": accuracy,
            "zero_rule_baseline": zero_rule_accuracy,
            "accuracy_lift": accuracy - zero_rule_accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "mcc": mcc,
            **class_acc,
        }
    
    def train_model(self, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols) -> Dict:
        """Train with properly temporally-split data."""
        
        logger.info("=" * 60)
        logger.info("V10: ICS_26 EXCLUDED + TEMPORAL SPLIT")
        logger.info(f"  Train: {len(X_train):,} samples (2014 to {self.train_end_date.date()})")
        logger.info(f"  Val:   {len(X_val):,} samples ({self.train_end_date.date()} to {self.val_end_date.date()})")
        logger.info(f"  Test:  {len(X_test):,} samples ({self.val_end_date.date()} to 2024)")
        logger.info("=" * 60)
        
        # Scale after split
        X_train, X_val, X_test = self.scale_data(X_train, X_val, X_test)
        
        model = AttentionLSTM(
            input_size=X_train.shape[-1],
            hidden_size=lstm_config.hidden_size if RUNNING_LOCAL else 128,
            num_layers=lstm_config.num_layers if RUNNING_LOCAL else 2,
            dropout=self.dropout,
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        ratio = len(X_train) / n_params
        logger.info(f"Model params: {n_params:,}, Samples/params: {ratio:.2f}x")
        
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(model, loss_fn, learning_rate=self.learning_rate, weight_decay=1e-4)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=self.batch_size
        )
        
        training_config.early_stopping_patience = self.early_stopping_patience
        history = trainer.train(train_loader, val_loader, epochs=self.epochs)
        
        logger.info("Evaluating on test set...")
        pred_classes = self.predict_batched(model, X_test)
        metrics = self.compute_metrics(y_test, pred_classes)
        cm = confusion_matrix(y_test, pred_classes)
        
        logger.info("\n" + "=" * 60)
        logger.info("V10 RESULTS (ICS_26 EXCLUDED)")
        logger.info("=" * 60)
        logger.info(f"Test Accuracy:        {metrics['accuracy']:.2%}")
        logger.info(f"Zero-Rule Baseline:   {metrics['zero_rule_baseline']:.2%}")
        logger.info(f"Accuracy Lift:        {metrics['accuracy_lift']:+.2%}")
        logger.info(f"F1 (macro):           {metrics['f1_macro']:.4f}")
        logger.info(f"MCC:                  {metrics['mcc']:.4f}")
        logger.info("=" * 60)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"\n{classification_report(y_test, pred_classes, target_names=['Down', 'Neutral', 'Up'])}")
        
        return {
            "model": "attention_lstm",
            "version": "v10",
            "train_end_date": str(self.train_end_date.date()),
            "val_end_date": str(self.val_end_date.date()),
            "n_features": X_train.shape[-1],
            "n_params": n_params,
            "n_stocks": len(self.stocks),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "samples_per_param": ratio,
            **metrics,
            "confusion_matrix": cm.tolist(),
            "best_val_loss": trainer.best_val_loss,
            "epochs_trained": len(history.get("history", [])),
        }
    
    def run(self):
        logger.info("=" * 80)
        logger.info("POOLED BASELINE V10: ICS_26 LEAK INVESTIGATION")
        logger.info("=" * 80)
        logger.info(f"EXCLUDED FEATURES: {EXCLUDED_FEATURES}")
        logger.info(f"  Train period: 2014 to {self.train_end_date.date()}")
        logger.info(f"  Val period:   {self.train_end_date.date()} to {self.val_end_date.date()}")
        logger.info(f"  Test period:  {self.val_end_date.date()} to 2024")
        logger.info("=" * 80)
        
        stock_data = self.load_all_stocks()
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = \
            self.prepare_temporal_pooled_data(stock_data)

        # Check for leaks
        check_for_leaks(X_train, y_train, feature_cols)
        
        del stock_data
        import gc
        gc.collect()
        
        with open(self.output_dir / "features_v10.txt", "w") as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        
        result = self.train_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
        
        results_df = pd.DataFrame([result])
        results_df.to_csv(self.output_dir / "results.csv", index=False)
        
        logger.info(f"\nResults saved to: {self.output_dir}")
        return result



def check_for_leaks(X_train, y_train, feature_cols):
    logger.info("🕵️ RUNNING LEAK DETECTION...")
    
    # 1. Convert back to DataFrame for easy analysis
    # Take a sample (first 50k rows) to save memory
    sample_size = min(50000, len(X_train))
    
    # We only need the LAST step of the sequence to check for direct correlation
    # Shape: (N, 20, Features) -> We take (N, 19, :) -> The most recent day in the seq
    last_day_features = X_train[:sample_size, -1, :] 
    
    df_check = pd.DataFrame(last_day_features, columns=feature_cols)
    df_check['TARGET_LABEL'] = y_train[:sample_size]
    
    # 2. correlation matrix
    correlations = df_check.corrwith(df_check['TARGET_LABEL'])
    
    # 3. Find the suspects
    suspects = correlations[abs(correlations) > 0.15]
    
    if len(suspects) > 0:
        logger.warning("🚨 LEAK DETECTED! The following features correlate too highly with the target:")
        logger.warning(suspects)
        logger.warning("Remove these features from indicators_v7.py")
    else:
        logger.info("✅ No direct correlation leaks detected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stocks", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--train-end", type=str, default="2022-01-01")
    parser.add_argument("--val-end", type=str, default="2023-06-01")
    parser.add_argument("--colab", action="store_true")
    args = parser.parse_args()
    
    stocks = EXTENDED_TICKERS[:args.stocks]
    
    experiment = PooledExperimentV10(
        stocks=stocks,
        train_end_date=args.train_end,
        val_end_date=args.val_end,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        dropout=args.dropout,
        learning_rate=args.lr,
        use_cache=not args.no_cache,
        colab_mode=args.colab,
    )
    
    experiment.run()


if __name__ == "__main__":
    main()
