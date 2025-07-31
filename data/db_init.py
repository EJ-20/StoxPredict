import sqlite3

def create_database(db_name="stocks.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            UNIQUE(ticker, date)
        )
    ''')

    conn.commit()
    conn.close()
    print(f"✅ Database '{db_name}' initialized with table 'stocks'")

if __name__ == "__main__":
    create_database()
