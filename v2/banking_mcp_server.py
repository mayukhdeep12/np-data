"""
banking_mcp_server.py — NEODesk Banking MCP Server
====================================================
Standalone FastMCP server exposing 11 banking tools.
This file is completely independent of the Streamlit app.

Run with MCP Inspector (visual testing UI):
    npx @modelcontextprotocol/inspector python banking_mcp_server.py

Run as plain stdio (for MCP clients / Claude Desktop):
    python banking_mcp_server.py

Run as HTTP server:
    python banking_mcp_server.py --transport streamable-http --port 8000
"""

import sys
import random
from datetime import datetime
from typing import Optional

from fastmcp import FastMCP

# ── Server Init ────────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="NEODesk Banking Server",
    instructions="""
    NEODesk Banking MCP Server provides core banking operations:
    account balances, card management, fraud detection, transaction history,
    fund transfers, spending analytics, and beneficiary management.
    All account data is simulated for demo purposes.
    """,
)

# ══════════════════════════════════════════════════════════════════════════════
# MOCK DATABASE  (in-memory, resets on server restart)
# ══════════════════════════════════════════════════════════════════════════════

ACCOUNTS = {
    "ACC001": {
        "holder":   "John Carter",
        "type":     "checking",
        "balance":  12450.75,
        "currency": "USD",
        "status":   "active",
        "opened":   "2019-03-15",
    },
    "ACC002": {
        "holder":   "John Carter",
        "type":     "savings",
        "balance":  38200.00,
        "currency": "USD",
        "status":   "active",
        "opened":   "2019-03-15",
    },
    "ACC003": {
        "holder":       "John Carter",
        "type":         "credit",
        "balance":      -2340.50,
        "currency":     "USD",
        "status":       "active",
        "credit_limit": 15000.00,
        "opened":       "2020-07-01",
    },
}

CARDS = {
    "CARD001": {
        "account_id":    "ACC001",
        "holder":        "John Carter",
        "last_four":     "4821",
        "type":          "Visa Debit",
        "status":        "active",
        "expiry":        "2027-08",
        "daily_limit":   2000.00,
        "contactless":   True,
        "international": False,
    },
    "CARD002": {
        "account_id":    "ACC003",
        "holder":        "John Carter",
        "last_four":     "9034",
        "type":          "Mastercard Credit",
        "status":        "active",
        "expiry":        "2026-11",
        "daily_limit":   5000.00,
        "contactless":   True,
        "international": True,
    },
}

TRANSACTIONS = [
    {"id": "TXN1001", "account": "ACC001", "date": "2025-01-28", "amount": -54.20,   "merchant": "Whole Foods",      "category": "Groceries",   "status": "completed"},
    {"id": "TXN1002", "account": "ACC001", "date": "2025-01-27", "amount": -12.99,   "merchant": "Netflix",          "category": "Streaming",   "status": "completed"},
    {"id": "TXN1003", "account": "ACC001", "date": "2025-01-27", "amount": -38.50,   "merchant": "Shell Gas",        "category": "Fuel",        "status": "completed"},
    {"id": "TXN1004", "account": "ACC001", "date": "2025-01-26", "amount": +2500.00, "merchant": "Employer Payroll", "category": "Income",      "status": "completed"},
    {"id": "TXN1005", "account": "ACC001", "date": "2025-01-25", "amount": -299.00,  "merchant": "Amazon",           "category": "Shopping",    "status": "completed"},
    {"id": "TXN1006", "account": "ACC001", "date": "2025-01-24", "amount": -8.50,    "merchant": "Starbucks",        "category": "Dining",      "status": "completed"},
    {"id": "TXN1007", "account": "ACC001", "date": "2025-01-23", "amount": -120.00,  "merchant": "Electric Company", "category": "Utilities",   "status": "completed"},
    {"id": "TXN1008", "account": "ACC001", "date": "2025-01-22", "amount": -45.00,   "merchant": "Uber",             "category": "Transport",   "status": "completed"},
    {"id": "TXN1009", "account": "ACC003", "date": "2025-01-28", "amount": -189.99,  "merchant": "Apple Store",      "category": "Electronics", "status": "completed"},
    {"id": "TXN1010", "account": "ACC003", "date": "2025-01-26", "amount": -67.40,   "merchant": "Restaurant XO",    "category": "Dining",      "status": "completed"},
    # Pre-seeded suspicious transactions for fraud demo
    {"id": "TXN1011", "account": "ACC001", "date": "2025-01-29", "amount": -4999.00, "merchant": "Unknown Vendor",   "category": "Unknown",     "status": "flagged", "flag": "Large unusual transaction"},
    {"id": "TXN1012", "account": "ACC001", "date": "2025-01-29", "amount": -1.00,    "merchant": "Online Store RU",  "category": "Shopping",    "status": "flagged", "flag": "Foreign merchant probe"},
]

BENEFICIARIES = {
    "BEN001": {"name": "Alice Johnson",   "bank": "Chase Bank",       "account_no": "****3421", "ifsc": "CHAS0001", "added": "2023-01-10"},
    "BEN002": {"name": "Bob Williams",    "bank": "Bank of America",  "account_no": "****8874", "ifsc": "BOFA0002", "added": "2023-06-22"},
    "BEN003": {"name": "Rent - LandLord", "bank": "Wells Fargo",      "account_no": "****5510", "ifsc": "WFBK0003", "added": "2022-11-05"},
}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Account Balance
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def get_account_balance(account_id: str) -> dict:
    """
    Get the current balance and details for a specific account.

    Args:
        account_id: Account ID — ACC001 (checking), ACC002 (savings), ACC003 (credit)
    """
    acc = ACCOUNTS.get(account_id.upper())
    if not acc:
        return {
            "success": False,
            "error": f"Account '{account_id}' not found.",
            "available_accounts": list(ACCOUNTS.keys()),
        }

    result = {
        "success":           True,
        "account_id":        account_id.upper(),
        "holder":            acc["holder"],
        "account_type":      acc["type"].title(),
        "balance":           acc["balance"],
        "formatted_balance": f"${acc['balance']:,.2f}",
        "currency":          acc["currency"],
        "status":            acc["status"].upper(),
        "opened_on":         acc["opened"],
        "as_of":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if acc["type"] == "credit":
        result["credit_limit"]     = acc.get("credit_limit", 0)
        result["available_credit"] = acc.get("credit_limit", 0) + acc["balance"]
        result["utilization_pct"]  = round(
            abs(acc["balance"]) / acc.get("credit_limit", 1) * 100, 1
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — All Accounts Summary
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def get_all_accounts() -> dict:
    """
    Get a summary of all accounts linked to the customer profile.
    Returns balances, types, and statuses for every account at once.
    """
    accounts     = []
    total_assets = 0.0

    for acc_id, acc in ACCOUNTS.items():
        accounts.append({
            "account_id":        acc_id,
            "type":              acc["type"].title(),
            "balance":           acc["balance"],
            "formatted_balance": f"${acc['balance']:,.2f}",
            "status":            acc["status"].upper(),
            "currency":          acc["currency"],
        })
        if acc["balance"] > 0:
            total_assets += acc["balance"]

    return {
        "success":        True,
        "customer":       "John Carter",
        "total_accounts": len(accounts),
        "total_assets":   f"${total_assets:,.2f}",
        "accounts":       accounts,
        "as_of":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Card Status
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def get_card_status(card_id: str) -> dict:
    """
    Get current status and details for a debit or credit card.

    Args:
        card_id: Card ID — CARD001 (Visa Debit) or CARD002 (Mastercard Credit)
    """
    card = CARDS.get(card_id.upper())
    if not card:
        return {
            "success": False,
            "error": f"Card '{card_id}' not found.",
            "available_cards": list(CARDS.keys()),
        }

    expiry_date = datetime.strptime(card["expiry"], "%Y-%m")
    is_expired  = expiry_date < datetime.now().replace(day=1)

    return {
        "success":                   True,
        "card_id":                   card_id.upper(),
        "card_type":                 card["type"],
        "holder":                    card["holder"],
        "last_four":                 f"**** **** **** {card['last_four']}",
        "linked_account":            card["account_id"],
        "status":                    card["status"].upper(),
        "expiry":                    card["expiry"],
        "is_expired":                is_expired,
        "daily_limit":               f"${card['daily_limit']:,.2f}",
        "contactless_enabled":       card["contactless"],
        "international_transactions": card["international"],
        "as_of":                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Block / Unblock / Freeze Card
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def update_card_status(card_id: str, action: str, reason: Optional[str] = None) -> dict:
    """
    Change the status of a card.

    Args:
        card_id: Card ID — CARD001 or CARD002
        action:  One of 'block', 'unblock', 'freeze'
        reason:  Optional reason (e.g. 'lost', 'stolen', 'suspected_fraud')
    """
    card = CARDS.get(card_id.upper())
    if not card:
        return {"success": False, "error": f"Card '{card_id}' not found."}

    valid_actions = {"block": "blocked", "unblock": "active", "freeze": "frozen"}
    if action.lower() not in valid_actions:
        return {
            "success": False,
            "error": f"Invalid action '{action}'. Use: block, unblock, or freeze.",
        }

    old_status   = card["status"]
    new_status   = valid_actions[action.lower()]
    card["status"] = new_status

    return {
        "success":         True,
        "card_id":         card_id.upper(),
        "card_type":       card["type"],
        "last_four":       card["last_four"],
        "previous_status": old_status.upper(),
        "new_status":      new_status.upper(),
        "reason":          reason or "Not specified",
        "reference":       f"CARD-CHG-{random.randint(10000, 99999)}",
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message":         f"Card {card_id.upper()} has been {new_status} successfully.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — Fraud Detection
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def detect_fraud(account_id: str) -> dict:
    """
    Run fraud detection analysis on an account.
    Returns a risk score (0–100), flagged transactions, and recommended actions.

    Args:
        account_id: Account ID to analyze — ACC001, ACC002, or ACC003
    """
    acc = ACCOUNTS.get(account_id.upper())
    if not acc:
        return {"success": False, "error": f"Account '{account_id}' not found."}

    acct_txns  = [t for t in TRANSACTIONS if t["account"] == account_id.upper()]
    flagged    = [t for t in acct_txns if t.get("status") == "flagged"]
    high_value = [t for t in acct_txns if abs(t["amount"]) > 1000 and t.get("status") != "flagged"]
    foreign    = [t for t in acct_txns if any(k in t["merchant"] for k in ["RU", "CN", "NG", "Unknown"])]

    risk_score   = 0
    risk_factors = []

    if flagged:
        risk_score += len(flagged) * 30
        risk_factors.append(f"{len(flagged)} flagged transaction(s) detected")
    if len(high_value) > 2:
        risk_score += 20
        risk_factors.append(f"{len(high_value)} high-value transactions in recent period")
    if foreign:
        risk_score += 15
        risk_factors.append("Transactions from unusual or foreign merchants detected")

    risk_score = min(risk_score, 100)

    if risk_score == 0:
        risk_level     = "LOW"
        recommendation = "No immediate action required. Account activity appears normal."
    elif risk_score <= 40:
        risk_level     = "MEDIUM"
        recommendation = "Monitor closely. Review flagged transactions with account holder."
    elif risk_score <= 70:
        risk_level     = "HIGH"
        recommendation = "Verify recent transactions. Consider temporary card freeze."
    else:
        risk_level     = "CRITICAL"
        recommendation = "Immediately block all cards. Contact account holder and escalate to fraud team."

    return {
        "success":              True,
        "account_id":           account_id.upper(),
        "holder":               acc["holder"],
        "risk_score":           risk_score,
        "risk_level":           risk_level,
        "risk_factors":         risk_factors or ["None detected"],
        "flagged_transactions": [
            {
                "transaction_id": t["id"],
                "date":           t["date"],
                "amount":         f"${t['amount']:,.2f}",
                "merchant":       t["merchant"],
                "reason":         t.get("flag", "Suspicious activity"),
            }
            for t in flagged
        ],
        "flagged_count":    len(flagged),
        "recommendation":   recommendation,
        "analysis_time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — Transaction History
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def get_transaction_history(
    account_id: str,
    limit: int = 10,
    category: Optional[str] = None,
) -> dict:
    """
    Retrieve recent transactions for an account.

    Args:
        account_id: Account ID — ACC001, ACC002, or ACC003
        limit:      Max transactions to return (default 10, max 50)
        category:   Optional filter — Groceries, Dining, Shopping, Fuel, etc.
    """
    if account_id.upper() not in ACCOUNTS:
        return {"success": False, "error": f"Account '{account_id}' not found."}

    txns = [t for t in TRANSACTIONS if t["account"] == account_id.upper()]
    if category:
        txns = [t for t in txns if t["category"].lower() == category.lower()]

    txns = txns[:min(limit, 50)]

    credits = sum(t["amount"] for t in txns if t["amount"] > 0)
    debits  = sum(t["amount"] for t in txns if t["amount"] < 0)

    return {
        "success":         True,
        "account_id":      account_id.upper(),
        "total_returned":  len(txns),
        "total_credits":   f"${credits:,.2f}",
        "total_debits":    f"${debits:,.2f}",
        "net":             f"${credits + debits:+,.2f}",
        "transactions": [
            {
                "transaction_id": t["id"],
                "date":           t["date"],
                "amount":         f"${t['amount']:+,.2f}",
                "merchant":       t["merchant"],
                "category":       t["category"],
                "status":         t["status"].upper(),
                "flagged":        t.get("status") == "flagged",
                "flag_reason":    t.get("flag"),
            }
            for t in txns
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 7 — Fund Transfer
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def transfer_funds(
    from_account_id: str,
    to_account_id: str,
    amount: float,
    note: Optional[str] = None,
) -> dict:
    """
    Transfer funds between two accounts.

    Args:
        from_account_id: Source account ID (e.g. ACC001)
        to_account_id:   Destination account ID (e.g. ACC002)
        amount:          Amount to transfer — must be positive
        note:            Optional transfer memo
    """
    from_acc = ACCOUNTS.get(from_account_id.upper())
    to_acc   = ACCOUNTS.get(to_account_id.upper())

    if not from_acc:
        return {"success": False, "error": f"Source account '{from_account_id}' not found."}
    if not to_acc:
        return {"success": False, "error": f"Destination account '{to_account_id}' not found."}
    if amount <= 0:
        return {"success": False, "error": "Transfer amount must be positive."}
    if from_account_id.upper() == to_account_id.upper():
        return {"success": False, "error": "Source and destination accounts cannot be the same."}
    if from_acc["balance"] < amount:
        return {
            "success": False,
            "error": f"Insufficient funds. Available: ${from_acc['balance']:,.2f}, Requested: ${amount:,.2f}",
        }

    from_acc["balance"] = round(from_acc["balance"] - amount, 2)
    to_acc["balance"]   = round(to_acc["balance"]   + amount, 2)

    return {
        "success":           True,
        "reference":         f"TRF{random.randint(1000000, 9999999)}",
        "from_account":      from_account_id.upper(),
        "from_type":         from_acc["type"].title(),
        "to_account":        to_account_id.upper(),
        "to_type":           to_acc["type"].title(),
        "amount":            f"${amount:,.2f}",
        "note":              note or "N/A",
        "from_new_balance":  f"${from_acc['balance']:,.2f}",
        "to_new_balance":    f"${to_acc['balance']:,.2f}",
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status":            "COMPLETED",
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 8 — Spending Analytics
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def get_spending_analytics(account_id: str) -> dict:
    """
    Analyze spending patterns with a category-wise breakdown.

    Args:
        account_id: Account ID — ACC001 or ACC003
    """
    if account_id.upper() not in ACCOUNTS:
        return {"success": False, "error": f"Account '{account_id}' not found."}

    txns = [t for t in TRANSACTIONS if t["account"] == account_id.upper() and t["amount"] < 0]

    totals: dict = {}
    for t in txns:
        totals[t["category"]] = totals.get(t["category"], 0) + abs(t["amount"])

    grand_total = sum(totals.values())

    breakdown = [
        {
            "category":   cat,
            "total":      f"${amt:,.2f}",
            "percentage": f"{amt / grand_total * 100:.1f}%" if grand_total else "0%",
        }
        for cat, amt in sorted(totals.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "success":           True,
        "account_id":        account_id.upper(),
        "period":            "Last 30 days",
        "total_spending":    f"${grand_total:,.2f}",
        "top_category":      breakdown[0]["category"] if breakdown else "N/A",
        "transaction_count": len(txns),
        "breakdown":         breakdown,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 9 — Get Beneficiaries
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def get_beneficiaries() -> dict:
    """
    Retrieve the list of saved beneficiaries for outbound transfers.
    """
    return {
        "success": True,
        "total":   len(BENEFICIARIES),
        "beneficiaries": [
            {
                "beneficiary_id": bid,
                "name":           b["name"],
                "bank":           b["bank"],
                "account":        b["account_no"],
                "ifsc":           b["ifsc"],
                "added_on":       b["added"],
            }
            for bid, b in BENEFICIARIES.items()
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 10 — Add Beneficiary
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def add_beneficiary(
    name: str,
    bank_name: str,
    account_number: str,
    ifsc_code: str,
) -> dict:
    """
    Add a new beneficiary for fund transfers.

    Args:
        name:           Full name of the beneficiary
        bank_name:      Name of beneficiary's bank
        account_number: Beneficiary's bank account number
        ifsc_code:      Bank IFSC / routing code
    """
    masked = "****" + account_number[-4:] if len(account_number) >= 4 else "****"
    new_id = f"BEN{str(len(BENEFICIARIES) + 1).zfill(3)}"

    BENEFICIARIES[new_id] = {
        "name":       name,
        "bank":       bank_name,
        "account_no": masked,
        "ifsc":       ifsc_code.upper(),
        "added":      datetime.now().strftime("%Y-%m-%d"),
    }

    return {
        "success":         True,
        "beneficiary_id":  new_id,
        "name":            name,
        "bank":            bank_name,
        "account":         masked,
        "ifsc":            ifsc_code.upper(),
        "added_on":        datetime.now().strftime("%Y-%m-%d"),
        "message":         f"Beneficiary '{name}' added successfully.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 11 — Mini Statement
# ══════════════════════════════════════════════════════════════════════════════
@mcp.tool()
def get_mini_statement(account_id: str) -> dict:
    """
    Get a mini statement with the last 5 transactions and current balance.

    Args:
        account_id: Account ID — ACC001, ACC002, or ACC003
    """
    acc = ACCOUNTS.get(account_id.upper())
    if not acc:
        return {"success": False, "error": f"Account '{account_id}' not found."}

    txns = [t for t in TRANSACTIONS if t["account"] == account_id.upper()][:5]

    return {
        "success":             True,
        "account_id":          account_id.upper(),
        "account_type":        acc["type"].title(),
        "holder":              acc["holder"],
        "current_balance":     f"${acc['balance']:,.2f}",
        "currency":            acc["currency"],
        "last_5_transactions": [
            {
                "date":        t["date"],
                "description": t["merchant"],
                "amount":      f"${t['amount']:+,.2f}",
                "status":      t["status"].upper(),
            }
            for t in txns
        ],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    transport = "stdio"
    port      = 8000

    args = sys.argv[1:]
    if "--transport" in args:
        transport = args[args.index("--transport") + 1]
    if "--port" in args:
        port = int(args[args.index("--port") + 1])

    print(f"[NEODesk MCP] Starting — transport={transport}, tools=11", file=sys.stderr)

    if transport == "streamable-http":
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")