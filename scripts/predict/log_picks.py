#!/usr/bin/env python3
"""
Log V2 model picks to Google Sheet.
Automatically clears duplicates for today before adding.
"""

import gspread
from google.oauth2.service_account import Credentials
from datetime import date
import os

CREDS_PATH = '/home/ubuntu/clawd/projects/nhl-model/credentials/google-service-account.json'
SHEET_ID = '1H23IR3wKuCQrifPuFJUceIMceWVpkIiv96IWSoaNbis'


def get_sheet():
    """Connect to Google Sheet"""
    creds = Credentials.from_service_account_file(
        CREDS_PATH,
        scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(SHEET_ID)
    return sheet.sheet1


def clear_today(ws):
    """Remove all rows for today's date"""
    today = date.today().strftime('%Y-%m-%d')
    
    rows = ws.get_all_values()
    rows_to_delete = []
    
    for i, row in enumerate(rows):
        if today in str(row):
            rows_to_delete.append(i + 1)
    
    # Delete from bottom up
    for row_num in reversed(rows_to_delete):
        ws.delete_rows(row_num)
    
    return len(rows_to_delete)


def log_pick(ws, game, pick, odds, model_prob, market_prob, edge):
    """Add a single pick to the sheet"""
    today = date.today().strftime('%Y-%m-%d')
    
    row = [
        today,
        game,
        pick,
        str(odds) if odds < 0 else f'+{odds}',
        f'{model_prob:.1f}%',
        f'{market_prob:.1f}%',
        f'+{edge:.1f}%' if edge > 0 else f'{edge:.1f}%',
        '',  # Result
        '',  # Profit
        '',  # CLV
    ]
    
    ws.append_row(row)
    return row


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
    
    # Add new picks
    for p in picks:
        row = log_pick(
            ws,
            p['game'],
            p['pick'],
            p['odds'],
            p['model'] * 100,
            p['market'] * 100,
            p['edge']
        )
        print(f"Logged: {p['game']} -> {p['pick']} ({p['edge']:+.1f}%)")
    
    print(f"\nTotal: {len(picks)} picks logged")


if __name__ == '__main__':
    # Test
    ws = get_sheet()
    deleted = clear_today(ws)
    print(f"Cleared {deleted} rows for today")
