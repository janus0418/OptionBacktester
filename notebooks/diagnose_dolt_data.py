#!/usr/bin/env python3
"""
Diagnostic script to investigate Dolt database option chain data.
This will help identify why queries for SPY on 2023-01-05 are failing.
"""

import sys
from pathlib import Path

# Add the code directory to path
code_path = Path(__file__).parent.parent / "code"
sys.path.insert(0, str(code_path))

from backtester.data.dolt_adapter import DoltAdapter
import pandas as pd

def main():
    print("=" * 80)
    print("DOLT DATABASE DIAGNOSTIC TOOL")
    print("=" * 80)

    # Initialize adapter
    db_path = '/Users/janussuk/Desktop/OptionsBacktester2/dolt_data/options'
    adapter = DoltAdapter(db_path)

    print("\n1. Checking database connection...")
    try:
        # Simple query to verify connection
        result = adapter.execute_query("SELECT COUNT(*) as count FROM option_chain LIMIT 1")
        print(f"   ✅ Database connected successfully")
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        return

    print("\n2. Checking table schema...")
    try:
        schema_query = "DESCRIBE option_chain"
        schema = adapter.execute_query(schema_query)
        print(f"   ✅ Table schema retrieved:")
        for _, row in schema.iterrows():
            print(f"      - {row['Field']}: {row['Type']}")
    except Exception as e:
        print(f"   ⚠️  Could not retrieve schema: {e}")

    print("\n3. Checking available symbols...")
    try:
        symbols_query = "SELECT DISTINCT act_symbol FROM option_chain ORDER BY act_symbol LIMIT 20"
        symbols = adapter.execute_query(symbols_query)
        print(f"   ✅ Found {len(symbols)} symbols (showing first 20):")
        print(f"      {', '.join(symbols['act_symbol'].tolist())}")
    except Exception as e:
        print(f"   ⚠️  Could not retrieve symbols: {e}")

    print("\n4. Checking SPY data availability...")
    try:
        spy_check = adapter.execute_query("""
            SELECT COUNT(*) as count, MIN(date) as min_date, MAX(date) as max_date
            FROM option_chain
            WHERE act_symbol = 'SPY'
        """)
        if len(spy_check) > 0:
            print(f"   ✅ SPY data found:")
            print(f"      - Total rows: {spy_check.iloc[0]['count']}")
            print(f"      - Date range: {spy_check.iloc[0]['min_date']} to {spy_check.iloc[0]['max_date']}")
        else:
            print(f"   ❌ No SPY data found")
    except Exception as e:
        print(f"   ⚠️  Could not check SPY data: {e}")

    print("\n5. Checking dates around 2023-01-05...")
    try:
        dates_query = """
            SELECT DISTINCT date
            FROM option_chain
            WHERE act_symbol = 'SPY'
              AND date >= '2023-01-01'
              AND date <= '2023-01-15'
            ORDER BY date
        """
        dates = adapter.execute_query(dates_query)
        if len(dates) > 0:
            print(f"   ✅ Found {len(dates)} trading dates in early January 2023:")
            for _, row in dates.iterrows():
                print(f"      - {row['date']}")
        else:
            print(f"   ⚠️  No SPY data found in early January 2023")
    except Exception as e:
        print(f"   ⚠️  Could not check dates: {e}")

    print("\n6. Checking specific date: 2023-01-05...")
    try:
        specific_query = """
            SELECT COUNT(*) as count,
                   MIN(expiration) as min_exp,
                   MAX(expiration) as max_exp
            FROM option_chain
            WHERE act_symbol = 'SPY'
              AND date = '2023-01-05'
        """
        specific = adapter.execute_query(specific_query)
        if len(specific) > 0 and specific.iloc[0]['count'] > 0:
            print(f"   ✅ Data found for SPY on 2023-01-05:")
            print(f"      - Total rows: {specific.iloc[0]['count']}")
            print(f"      - Expiration range: {specific.iloc[0]['min_exp']} to {specific.iloc[0]['max_exp']}")
        else:
            print(f"   ⚠️  No data found for SPY on 2023-01-05")
    except Exception as e:
        print(f"   ⚠️  Could not check specific date: {e}")

    print("\n7. Testing DTE calculation for 2023-01-05...")
    try:
        from datetime import datetime, timedelta

        target_date = datetime(2023, 1, 5)
        min_dte = 7
        max_dte = 60

        min_expiry = target_date + timedelta(days=min_dte)
        max_expiry = target_date + timedelta(days=max_dte)

        print(f"   Target date: {target_date.date()}")
        print(f"   DTE range: [{min_dte}, {max_dte}]")
        print(f"   Expiration range: {min_expiry.date()} to {max_expiry.date()}")

        dte_query = f"""
            SELECT COUNT(*) as count,
                   MIN(expiration) as min_exp,
                   MAX(expiration) as max_exp
            FROM option_chain
            WHERE act_symbol = 'SPY'
              AND date = '2023-01-05'
              AND expiration >= '{min_expiry.date()}'
              AND expiration <= '{max_expiry.date()}'
        """
        dte_result = adapter.execute_query(dte_query)

        if len(dte_result) > 0 and dte_result.iloc[0]['count'] > 0:
            print(f"   ✅ Found {dte_result.iloc[0]['count']} rows with DTE filter")
            print(f"      - Expiration range: {dte_result.iloc[0]['min_exp']} to {dte_result.iloc[0]['max_exp']}")
        else:
            print(f"   ⚠️  No data found with DTE filter")

            # Try without DTE filter
            no_dte_query = """
                SELECT DISTINCT expiration
                FROM option_chain
                WHERE act_symbol = 'SPY'
                  AND date = '2023-01-05'
                ORDER BY expiration
                LIMIT 10
            """
            expirations = adapter.execute_query(no_dte_query)
            if len(expirations) > 0:
                print(f"   📊 Available expirations on 2023-01-05 (showing first 10):")
                for _, row in expirations.iterrows():
                    exp_date = pd.to_datetime(row['expiration'])
                    dte = (exp_date - target_date).days
                    print(f"      - {row['expiration']} (DTE: {dte})")
    except Exception as e:
        print(f"   ⚠️  Error testing DTE calculation: {e}")

    print("\n8. Sample data from 2023-01-05...")
    try:
        sample_query = """
            SELECT *
            FROM option_chain
            WHERE act_symbol = 'SPY'
              AND date = '2023-01-05'
            LIMIT 5
        """
        sample = adapter.execute_query(sample_query)
        if len(sample) > 0:
            print(f"   ✅ Sample rows (showing first 5):")
            print(sample.to_string())
        else:
            print(f"   ⚠️  No sample data available")
    except Exception as e:
        print(f"   ⚠️  Could not retrieve sample data: {e}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    adapter.close()

if __name__ == "__main__":
    main()
