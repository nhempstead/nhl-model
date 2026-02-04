#!/usr/bin/env python3
"""
Log V2 model picks to Google Sheet.
Matches the sheet format: #, Date, Game, Pick, Open, Close, Model%, Mkt%, Edge, Conf, Units
"""

import gspread
from google.oauth2.service_account import Credentials
from datetime import date
import os

CREDS_PATH = '/home/ubuntu/clawd/projects/nhl-model/credentials/google-service-account.json'
SHEET_ID = '1H23IR3wKuCQrifPuFJUceIMceWVpkIiv96IWSoaNbis'
DATA_START_ROW = 8  # First data row (after header at row 6, blank row 7)


def get_sheet():
    """Connect to Google Sheet"""
    creds = Credentials.from_service_account_file(
        CREDS_PATH,
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(SHEET_ID)
    return sheet.sheet1


def get_next_row_number(ws):
    """Get the next row number (#) for new entries"""
    col_a = ws.col_values(1)
    max_num = 0
    for val in col_a:
        try:
            num = int(val)
            if num > max_num:
                max_num = num
        except:
            pass
    return max_num + 1


def find_insert_row(ws):
    """Find the next empty row in the data section"""
    col_a = ws.col_values(1)
    # Find last row with data
    last_row = len(col_a)
    # Insert after last data row, but minimum is DATA_START_ROW
    return max(last_row + 1, DATA_START_ROW)


def clear_today(ws):
    """Remove all rows for today's date"""
    today = date.today().strftime('%Y-%m-%d')
    
    all_data = ws.get_all_values()
    rows_to_delete = []
    
    # Check column B (Date) for today's date, starting from row 8
    for i, row in enumerate(all_data[7:], start=8):  # Start from row 8 (index 7)
        if len(row) > 1 and row[1] == today:
            rows_to_delete.append(i)
    
    # Delete from bottom up to preserve row indices
    for row_num in reversed(rows_to_delete):
        ws.delete_rows(row_num)
    
    return len(rows_to_delete)


def get_confidence_emoji(edge):
    """Get confidence indicator based on edge size"""
    if edge >= 10:
        return '🔥🔥'
    elif edge >= 5:
        return '🔥'
    else:
        return ''


def calculate_kelly_units(model_prob, odds):
    """Calculate quarter-Kelly units"""
    p = model_prob
    q = 1 - p
    
    # Convert American odds to decimal
    if odds > 0:
        decimal_odds = (odds / 100) + 1
    else:
        decimal_odds = (100 / abs(odds)) + 1
    
    b = decimal_odds - 1
    kelly = (b * p - q) / b
    
    # Quarter Kelly, capped at 5 units
    units = min(max(kelly / 4 * 100, 0.5), 5)
    return round(units, 1)


def log_pick(ws, row_num, game, pick, odds, model_prob, market_prob, edge):
    """Add a single pick to the sheet at the correct position"""
    today = date.today().strftime('%Y-%m-%d')
    
    # Format odds string
    odds_str = f'+{odds}' if odds > 0 else str(odds)
    
    # Get confidence and units
    conf = get_confidence_emoji(edge)
    units = calculate_kelly_units(model_prob / 100, odds)
    
    # Row format: #, Date, Game, Pick, Open, Close, Model%, Mkt%, Edge, Conf, Units
    row = [
        row_num,                          # A: #
        today,                            # B: Date
        game,                             # C: Game
        pick,                             # D: Pick
        odds_str,                         # E: Open
        '',                               # F: Close
        f'{model_prob:.1f}%',             # G: Model%
        f'{market_prob:.1f}%',            # H: Mkt%
        f'+{edge:.1f}%' if edge > 0 else f'{edge:.1f}%',  # I: Edge
        conf,                             # J: Conf
        units,                            # K: Units
    ]
    
    # Find the row to insert
    insert_row = find_insert_row(ws)
    
    # Write to specific row starting at column A
    ws.update(values=[row], range_name=f'A{insert_row}:K{insert_row}')
    
    return insert_row


def log_picks(picks):
    """
    Log multiple picks, clearing today's duplicates first.
    
    picks: list of dicts with keys: game, pick, odds, model, market, edge
    """
    ws = get_sheet()
    
    # Clear existing picks for today
    deleted = clear_today(ws)
    if deleted > 0:
        print(f"Cleared {deleted} existing rows for today")
    
    # Get starting row number
    row_num = get_next_row_number(ws)
    
    # Add new picks
    for p in picks:
        edge = p['edge']
        model_pct = p['model'] * 100
        market_pct = p['market'] * 100
        
        log_pick(
            ws,
            row_num,
            p['game'],
            p['pick'],
            p['odds'],
            model_pct,
            market_pct,
            edge
        )
        
        conf = get_confidence_emoji(edge)
        units = calculate_kelly_units(p['model'], p['odds'])
        print(f"#{row_num}: {p['game']} → {p['pick']} ({p['odds']:+d}) | {edge:+.1f}% {conf} | {units}u")
        row_num += 1
    
    print(f"\n✅ Logged {len(picks)} picks to sheet")


if __name__ == '__main__':
    # Test - clear today's entries
    ws = get_sheet()
    deleted = clear_today(ws)
    print(f"Cleared {deleted} rows for today")
