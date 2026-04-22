# Prediction Market Bot

## Quick Navigation Fix (PowerShell CD Error)

**Problem**: `cd prediction_market_bot` auto-corrects to uppercase, fails.

**Solutions**:
1. **Quotes**: `cd "prediction_market_bot"`
2. **Full path**: `cd c:/Users/igli/prediction_market_bot`
3. **Disable auto-correct**:
   - Temp: `Set-PSReadlineOption -PredictionSource None`
   - Permanent: Add to `$PROFILE`: `notepad \$PROFILE`, append `Set-PSReadlineOption -PredictionSource None`

## Project Files
- `main.py`: Entry point
- `config.py`: Settings
- `polymarket.py`: API integration
- Others: Strategies (mean_reversion, news_sentiment), managers (risk, order)

Run: `python main.py`

