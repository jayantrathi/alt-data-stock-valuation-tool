# Signal Discovery Engine

I built this to answer a question I was curious about: for a given stock, can you find external signals : macro data, commodity prices, competitor stock moves, search trends, that actually lead its price? And if you can, do those relationships hold up over time, or do they fall apart?

## What it does

For any ticker, it pulls a bunch of data and lines it up against the stock's future returns:

- Price and forward returns from Yahoo Finance
- Macro data from FRED (rates, inflation, unemployment, VIX, oil, the dollar, etc.)
- The stock's sector ETF and the S&P 500
- Competitor stock prices — found automatically by figuring out the company's industry and ranking peers by correlation (dynamic, not hardcoded)
- Google Trends search interest
- Industry-specific news data — it reads the company's business description and pulls things that matter for that business (an airline gets jet fuel and air-travel demand, a bank gets the yield curve, and so on)

Then for every signal it checks: does it correlate with future returns? At what time lag? Is it statistically significant? And most importantly — is the correlation *stable* over time, or does it flip around?

## What I found

- Most signals are noise. For a typical large-cap, 50–90% of what I tested didn't clear significance.
- The ones that did clear it were usually unstable — the correlation swings positive to negative depending on the time period, which makes them useless for actually predicting anything.
- What drives a stock is specific to that stock. Apple and Google are mostly moved by macro stuff. Southwest (LUV) was mostly moved by other airlines' stock prices, about 3 weeks ahead.
- Some "obvious" relationships just aren't there. I expected jet fuel to clearly predict Southwest. It didn't — airlines hedge fuel and the market prices it in fast.
- Correlations top out around 0.3–0.4, which is too weak and too unstable to trade on.

## Example output

Each run produces a report like this:

![AAPL](images/AAPL_signals.png)
![LUV](images/LUV_signals.png)

## Running it

​```bash
pip install -r requirements.txt
cp .env.example .env      # then add your free FRED key
python data_pipeline.py   # build the data for a ticker
python signal_engine.py   # analyse it and make the report
​```

## Files

- `data_pipeline.py` — pulls and combines all the data for a ticker
- `industry_signals.py` — the industry-specific data layer
- `signal_engine.py` — does the correlation, significance and stability analysis and the plots

## Honest limitations

- It works on daily data. Lead-lag effects are stronger intraday and mostly fade at daily frequency, so this is testing on the hard setting.
- It tests one stock's absolute returns. A better version would test performance *relative to a basket of similar stocks*, which cancels out a lot of market-wide noise. That's the next thing I'd change.
- With this many signals and lags, some "significant" results happen by chance — which is exactly why I care more about stability than raw significance.

Built with Python, pandas, NumPy, SciPy, matplotlib, yfinance, and the FRED API.
