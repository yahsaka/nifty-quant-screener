def find_cached_ticker(company_name, cache_dir):
    """Token-based matching with a Master Override Dictionary for edge cases."""
    original_name = str(company_name).strip().upper()
    
    # ==========================================
    # 1. THE MASTER OVERRIDE DICTIONARY
    # Add any stubborn, mismatched, or weirdly formatted broker names here.
    # Format: "EXACT BROKER EXPORT NAME": "YAHOO_TICKER.NS"
    # ==========================================
    MASTER_MAPPING = {
        # Tricky ETFs
        "NIP IND ETF LIQUID BEES": "LIQUIDBEES.NS",
        "LIQUID BEES": "LIQUIDBEES.NS",
        "NIPPON INDIA ETF JUNIOR BEES": "JUNIORBEES.NS",
        "NIPPON INDIA ETF NIFTY BEES": "NIFTYBEES.NS",
        "NIPPON INDIA ETF BANK BEES": "BANKBEES.NS",
        "NIPPON INDIA ETF GOLD BEES": "GOLDBEES.NS",
        
        # Stubborn Corporate Names (from your errors)
        "H.G.INFRA ENGINEERING LTD": "HGINFRA.NS",
        "CEIGALL INDIA LIMITED": "CEIGALL.NS",
        "JKUMAR INFR.LTD.": "JKIL.NS",
        "AMARA RAJA ENERGY MOB LTD": "ARE&M.NS",
        "APOLLO TYRES LTD": "APOLLOTYRE.NS",
        "COAL INDIA LTD": "COALINDIA.NS",
        "JINDAL SAW LIMITED": "JINDALSAW.NS",
        "OIL AND NATURAL GAS CORP.": "ONGC.NS",
        "SUN TV NETWORK LIMITED": "SUNTV.NS",
        
        # Add more as you discover them in your broker files!
    }

    # Instant return if the exact name is in our master dictionary
    if original_name in MASTER_MAPPING:
        return MASTER_MAPPING[original_name]


    # ==========================================
    # 2. Algorithmic Fallback (For everything else)
    # ==========================================
    def normalize_name(name):
        name = str(name).upper().replace("&", "AND").replace(".", "")
        name = re.sub(r"[^A-Z0-9\s]", " ", name)
        stopwords = {"LTD", "LIMITED", "INC", "CORP", "CORPORATION", "CO", "COMPANY", "L"}
        return [w for w in name.split() if w not in stopwords]

    broker_tokens = normalize_name(original_name)
    if not broker_tokens:
        return f"{original_name}.NS"

    nse_csv = "data/nifty500_tickers.csv"
    if os.path.exists(nse_csv):
        try:
            nse_df = pd.read_csv(nse_csv)
            symbols_set = set(nse_df["Symbol"].dropna().astype(str).str.upper())
            
            if original_name in symbols_set:
                return f"{original_name}.NS"
                
            if broker_tokens[0] in symbols_set:
                return f"{broker_tokens[0]}.NS"
            
            if "Company Name" in nse_df.columns:
                best_match_symbol = None
                best_score = 0.0
                
                for _, row in nse_df.iterrows():
                    nse_name = str(row["Company Name"])
                    nse_symbol = str(row["Symbol"]).upper()
                    nse_tokens = normalize_name(nse_name)
                    
                    if not nse_tokens: continue
                        
                    set_b = set(broker_tokens)
                    set_n = set(nse_tokens)
                    
                    overlap = len(set_b.intersection(set_n))
                    score = overlap / min(len(set_b), len(set_n))
                    
                    if broker_tokens[0] == nse_tokens[0] and len(broker_tokens[0]) >= 3:
                        score += 0.30
                        
                    if score > best_score:
                        best_score = score
                        best_match_symbol = nse_symbol

                if best_score >= 0.75:
                    return f"{best_match_symbol}.NS"
        except Exception:
            pass

    if os.path.exists(cache_dir):
        cached_files = [f.replace(".parquet", "") for f in os.listdir(cache_dir) if f.endswith(".parquet")]
        if original_name in cached_files:
            return original_name
            
        core_name_no_space = "".join(broker_tokens)
        for ticker in cached_files:
            sym = ticker.replace(".NS", "").upper()
            if sym == core_name_no_space or (len(sym) >= 4 and sym in core_name_no_space):
                return ticker

    guessed_ticker = broker_tokens[0]
    if len(guessed_ticker) < 3 and len(broker_tokens) > 1:
        guessed_ticker = f"{broker_tokens[0]}{broker_tokens[1]}"
        
    return f"{guessed_ticker}.NS"
