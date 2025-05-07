-- 0. Győződj meg róla, hogy a trading_db adatbázis létezik
CREATE DATABASE IF NOT EXISTS trading_db;
USE trading_db;

-- 1. market_data tábla létrehozása (piaci adatok, indikátorok, sentiment)
CREATE TABLE IF NOT EXISTS market_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10),
    timestamp DATETIME,
    price DECIMAL(18,8),
    open_price DECIMAL(18,8),
    high DECIMAL(18,8),
    low DECIMAL(18,8),
    bid_price DECIMAL(18,8),
    ask_price DECIMAL(18,8),
    volume DECIMAL(18,8),
    sentiment VARCHAR(20),
    rsi FLOAT,
    macd FLOAT,
    macd_signal FLOAT,
    macd_diff FLOAT,
    stoch FLOAT,
    stoch_signal FLOAT,
    sma FLOAT,
    ema FLOAT,
    bb_hband FLOAT,
    bb_lband FLOAT,
    bb_mavg FLOAT,
    atr FLOAT,
    adx FLOAT,
    obv FLOAT,
    supertrend FLOAT,
    supertrend_trend VARCHAR(10),
    alligator_jaw FLOAT,
    alligator_teeth FLOAT,
    alligator_lips FLOAT,
    pivot FLOAT,
    r1 FLOAT,
    s1 FLOAT,
    r2 FLOAT,
    s2 FLOAT,
    r3 FLOAT,
    s3 FLOAT,
    fib_0 FLOAT,
    fib_236 FLOAT,
    fib_382 FLOAT,
    fib_50 FLOAT,
    fib_618 FLOAT,
    fib_786 FLOAT,
    fib_1 FLOAT,
    UNIQUE KEY unique_symbol_time (symbol, timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2. technical_indicators tábla létrehozása
CREATE TABLE IF NOT EXISTS technical_indicators (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    symbol VARCHAR(10),
    rsi FLOAT,
    macd FLOAT,
    macd_signal FLOAT,
    macd_diff FLOAT,
    stoch FLOAT,
    stoch_signal FLOAT,
    sma FLOAT,
    ema FLOAT,
    bb_hband FLOAT,
    bb_lband FLOAT,
    bb_mavg FLOAT,
    atr FLOAT,
    adx FLOAT,
    obv FLOAT,
    supertrend FLOAT,
    supertrend_trend VARCHAR(10),
    alligator_jaw FLOAT,
    alligator_teeth FLOAT,
    alligator_lips FLOAT,
    pivot FLOAT,
    r1 FLOAT,
    s1 FLOAT,
    r2 FLOAT,
    s2 FLOAT,
    r3 FLOAT,
    s3 FLOAT,
    fib_0 FLOAT,
    fib_236 FLOAT,
    fib_382 FLOAT,
    fib_50 FLOAT,
    fib_618 FLOAT,
    fib_786 FLOAT,
    fib_1 FLOAT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. model_predictions tábla létrehozása
CREATE TABLE IF NOT EXISTS model_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10),
    timestamp DATETIME,
    prediction VARCHAR(20),
    confidence FLOAT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. market_price_history tábla létrehozása
CREATE TABLE IF NOT EXISTS market_price_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE,
    symbol VARCHAR(10),
    open_price DECIMAL(18,8),
    close_price DECIMAL(18,8),
    high DECIMAL(18,8),
    low DECIMAL(18,8),
    volume DECIMAL(18,8)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. order_book tábla létrehozása
CREATE TABLE IF NOT EXISTS order_book (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    symbol VARCHAR(10),
    top_bid_price DECIMAL(18,8),
    top_bid_volume DECIMAL(18,8),
    top_ask_price DECIMAL(18,8),
    top_ask_volume DECIMAL(18,8)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 6. trade_logs tábla létrehozása
CREATE TABLE IF NOT EXISTS trade_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10),
    order_timestamp DATETIME,
    order_side VARCHAR(10),
    quantity DECIMAL(18,8),
    prediction VARCHAR(20),
    confidence FLOAT,
    reason TEXT,
    additional_info TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 7. symbols tábla létrehozása
CREATE TABLE IF NOT EXISTS symbols (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 8. market_news tábla létrehozása
CREATE TABLE IF NOT EXISTS market_news (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    symbol VARCHAR(10),
    title VARCHAR(255),
    content TEXT,
    published DATETIME
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 9. integrated_market_data tábla létrehozása
CREATE TABLE IF NOT EXISTS integrated_market_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    symbol VARCHAR(10),
    price DECIMAL(18,8),
    open_price DECIMAL(18,8),
    high DECIMAL(18,8),
    low DECIMAL(18,8),
    bid_price DECIMAL(18,8),
    ask_price DECIMAL(18,8),
    volume DECIMAL(18,8),
    sentiment VARCHAR(20),
    rsi FLOAT,
    macd FLOAT,
    macd_signal FLOAT,
    macd_diff FLOAT,
    stoch FLOAT,
    stoch_signal FLOAT,
    sma FLOAT,
    ema FLOAT,
    bb_hband FLOAT,
    bb_lband FLOAT,
    bb_mavg FLOAT,
    atr FLOAT,
    adx FLOAT,
    obv FLOAT,
    supertrend FLOAT,
    supertrend_trend VARCHAR(10),
    alligator_jaw FLOAT,
    alligator_teeth FLOAT,
    alligator_lips FLOAT,
    pivot FLOAT,
    r1 FLOAT,
    s1 FLOAT,
    r2 FLOAT,
    s2 FLOAT,
    r3 FLOAT,
    s3 FLOAT,
    fib_0 FLOAT,
    fib_236 FLOAT,
    fib_382 FLOAT,
    fib_50 FLOAT,
    fib_618 FLOAT,
    fib_786 FLOAT,
    fib_1 FLOAT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
