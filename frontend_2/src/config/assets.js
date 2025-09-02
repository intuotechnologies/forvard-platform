// src/config/assets.js - SINCRONIZZATO con il backend init_users.py

// Asset BASE - IDENTICI a quelli nel backend Python
// Questi sono gli asset che gli utenti base possono vedere/scaricare
export const BASE_ASSETS = {
  stocks: [
    'AAPL', 'ADBE', 'AMD', 'AMZN', 'AXP', 'BA', 'CAT', 'COIN', 'CSCO', 'DIS',
    'EBAY', 'GE', 'GOOGL', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO',
    'MCD', 'META', 'MMM', 'MSFT', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PG', 'PM',
    'PYPL', 'SHOP', 'SNAP', 'SPOT', 'TSLA', 'UBER', 'V', 'WMT', 'XOM', 'ZM'
  ],
  /*
  etf: [
    'AGG', 'BND', 'GLD', 'SLV', 'SUSA', 'EFIV', 'ESGV', 'ESGU', 'AFTY', 'MCHI',
    'EWH', 'EEM', 'IEUR', 'VGK', 'FLCH', 'EWJ', 'NKY', 'EWZ', 'EWC', 'EWU',
    'EWI', 'EWP', 'ACWI', 'IOO', 'GWL', 'VEU', 'IJH', 'MDY', 'IVOO', 'IYT',
    'XTN', 'XLI', 'XLU', 'VPU', 'SPSM', 'IJR', 'VIOO', 'QQQ', 'ICLN', 'ARKK',
    'SPLG', 'SPY', 'VOO', 'IYY', 'VTI', 'DIA'
  ],
  */
  forex: ['EURUSD', 'GBPUSD', 'AUDUSD', 'CADUSD', 'JPYUSD'],
  futures: ['ES', 'CL', 'GC', 'NG']
};

// Asset categories per il dashboard
export const ASSET_CATEGORIES = {
  stocks: {
    name: "Stocks",
    symbols: BASE_ASSETS.stocks
  },
  /*
  etf: {
    name: "ETFs", 
    symbols: BASE_ASSETS.etf
  },
  */
  forex: {
    name: "Forex",
    symbols: BASE_ASSETS.forex
  },
  futures: {
    name: "Futures",
    symbols: BASE_ASSETS.futures
  }
};

// Utility functions
export const validateAssetAccess = (symbol, assetType) => {
  if (!symbol || !assetType) return false;
  
  const normalizedSymbol = symbol.toUpperCase();
  const normalizedAssetType = assetType.toLowerCase();
  
  if (!BASE_ASSETS[normalizedAssetType]) return false;
  
  return BASE_ASSETS[normalizedAssetType].includes(normalizedSymbol);
};

export const getAssetsByType = (assetType) => {
  return BASE_ASSETS[assetType.toLowerCase()] || [];
};

export const getTotalAssetsCount = () => {
  return Object.values(BASE_ASSETS).reduce((total, assets) => total + assets.length, 0);
};

console.log('Asset configuration loaded:');
console.log(`- Stocks: ${BASE_ASSETS.stocks.length}`);
/* console.log(`- ETFs: ${BASE_ASSETS.etf?.length || 0}`); */
console.log(`- Forex: ${BASE_ASSETS.forex.length}`);
console.log(`- Futures: ${BASE_ASSETS.futures.length}`);
console.log(`- Total: ${getTotalAssetsCount()} assets`);