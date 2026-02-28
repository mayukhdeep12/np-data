"""
NEODesk Banking MCP Server
--------------------------
Run with:
    python banking_mcp_server.py           # stdio mode (for MCP clients)
    python banking_mcp_server.py --http    # HTTP mode (for MCP Inspector)

MCP Inspector:
    npx @modelcontextprotocol/inspector python banking_mcp_server.py
"""

import sys
import random
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from fastmcp import FastMCP

# ─────────────────────────────────────────────────────────────────────────────
# Server init
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="NEODesk Banking MCP",
    instructions="""
    You are connected to the NEODesk core banking system.
    Use these tools to fetch real account data, detect fraud,
    manage cards, process transactions and more.
    Never expose raw internal IDs or system errors to end users.
    """,
)

# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATABASE  (replace with real DB/API calls in production)
# ─────────────────────────────────────────────────────────────────────────────

MOCK_ACCOUNTS = {
    "ACC001": {
        "customer_name": "Arjun Sharma",
        "savings_balance": 124530.00,
        "current_balance": 382900.00,
        "fd_balance": 500000.00,
        "currency": "INR",
        "status": "ACTIVE",
        "kyc": "VERIFIED",
        "branch": "Mumbai - Andheri West",
        "ifsc": "NEOB0001234",
    },
    "ACC002": {
        "customer_name": "Priya Mehta",
        "savings_balance": 34200.00,
        "current_balance": 0.00,
        "fd_balance": 200000.00,
        "currency": "INR",
        "status": "ACTIVE",
        "kyc": "VERIFIED",
        "branch": "Delhi - Connaught Place",
        "ifsc": "NEOB0005678",
    },
}

MOCK_CARDS = {
    "ACC001": [
        {
            "card_number_masked": "****-****-****-4521",
            "card_type": "DEBIT",
            "network": "Visa",
            "status": "ACTIVE",
            "daily_limit": 50000,
            "used_today": 2340,
            "expiry": "08/27",
        },
        {
            "card_number_masked": "****-****-****-8873",
            "card_type": "CREDIT",
            "network": "Mastercard",
            "status": "ACTIVE",
            "credit_limit": 80000,
            "available_credit": 45230,
            "due_date": "05 Jul 2025",
            "minimum_due": 2400,
            "expiry": "03/28",
        },
    ],
    "ACC002": [
        {
            "card_number_masked": "****-****-****-7712",
            "card_type": "DEBIT",
            "network": "RuPay",
            "status": "BLOCKED",
            "daily_limit": 25000,
            "used_today": 0,
            "expiry": "11/26",
        },
    ],
}

MOCK_TRANSACTIONS = {
    "ACC001": [
        {"date": "2025-06-14", "description": "Zomato",           "amount": -340,   "category": "Food",        "ref": "TXN8821"},
        {"date": "2025-06-13", "description": "Amazon",           "amount": -2199,  "category": "Shopping",    "ref": "TXN8799"},
        {"date": "2025-06-12", "description": "NEFT from Rahul",  "amount": +10000, "category": "Transfer-In", "ref": "TXN8754"},
        {"date": "2025-06-11", "description": "Electricity Bill", "amount": -1450,  "category": "Utilities",   "ref": "TXN8712"},
        {"date": "2025-06-10", "description": "ATM Withdrawal",   "amount": -5000,  "category": "Cash",        "ref": "TXN8690"},
        {"date": "2025-06-09", "description": "Salary Credit",    "amount": +85000, "category": "Income",      "ref": "TXN8645"},
        {"date": "2025-06-08", "description": "Netflix",          "amount": -649,   "category": "OTT",         "ref": "TXN8612"},
        {"date": "2025-06-07", "description": "Swiggy",           "amount": -520,   "category": "Food",        "ref": "TXN8598"},
        {"date": "2025-06-06", "description": "Petrol Pump",      "amount": -3200,  "category": "Fuel",        "ref": "TXN8570"},
        {"date": "2025-06-05", "description": "Medical Store",    "amount": -890,   "category": "Health",      "ref": "TXN8541"},
    ],
}

MOCK_LOANS = {
    "personal": {"max_amount": 500000,   "rate": 10.5, "max_tenure_months": 60},
    "home":     {"max_amount": 5000000,  "rate": 8.75, "max_tenure_months": 240},
    "car":      {"max_amount": 1000000,  "rate": 9.25, "max_tenure_months": 84},
    "education":{"max_amount": 1500000,  "rate": 9.0,  "max_tenure_months": 120},
    "gold":     {"max_amount": 300000,   "rate": 7.5,  "max_tenure_months": 24},
}

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — Account Balance
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_account_balance(account_id: str, account_type: str = "all") -> dict:
    """
    Retrieve the current balance for a customer account.

    Args:
        account_id:   Customer account ID (e.g. ACC001)
        account_type: One of 'savings', 'current', 'fd', or 'all'

    Returns:
        Balance details with currency and account status.
    """
    acc = MOCK_ACCOUNTS.get(account_id.upper())
    if not acc:
        return {"error": f"Account '{account_id}' not found."}

    if acc["status"] != "ACTIVE":
        return {"error": f"Account '{account_id}' is {acc['status']}. Please visit the branch."}

    at = account_type.lower()
    result = {
        "account_id": account_id.upper(),
        "customer_name": acc["customer_name"],
        "currency": acc["currency"],
        "as_of": datetime.now().strftime("%d %b %Y, %H:%M"),
        "status": acc["status"],
    }

    if at in ("savings", "all"):
        result["savings_balance"] = f"₹{acc['savings_balance']:,.2f}"
    if at in ("current", "all"):
        result["current_balance"] = f"₹{acc['current_balance']:,.2f}"
    if at in ("fd", "all"):
        result["fd_balance"] = f"₹{acc['fd_balance']:,.2f}"

    if at not in ("savings", "current", "fd", "all"):
        return {"error": f"Unknown account_type '{account_type}'. Use: savings, current, fd, all."}

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — Recent Transactions
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_recent_transactions(
    account_id: str,
    limit: int = 5,
    category: Optional[str] = None,
) -> dict:
    """
    Fetch recent transactions for an account, optionally filtered by category.

    Args:
        account_id: Customer account ID
        limit:      Number of transactions to return (1–10)
        category:   Optional filter e.g. 'Food', 'Shopping', 'Income', 'Transfer-In'

    Returns:
        List of transactions with date, description, amount, category, and reference.
    """
    txns = MOCK_TRANSACTIONS.get(account_id.upper(), [])
    if not txns:
        return {"error": f"No transactions found for account '{account_id}'."}

    if category:
        txns = [t for t in txns if t["category"].lower() == category.lower()]

    txns = txns[: max(1, min(limit, 10))]

    formatted = []
    for t in txns:
        sign = "+" if t["amount"] > 0 else ""
        formatted.append({
            "date":        t["date"],
            "description": t["description"],
            "amount":      f"{sign}₹{abs(t['amount']):,.0f}",
            "category":    t["category"],
            "reference":   t["ref"],
        })

    return {
        "account_id": account_id.upper(),
        "count": len(formatted),
        "transactions": formatted,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — Card Status
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_card_status(account_id: str, card_type: Optional[str] = None) -> dict:
    """
    Retrieve the status and details of cards linked to an account.

    Args:
        account_id: Customer account ID
        card_type:  Optional filter — 'DEBIT' or 'CREDIT'

    Returns:
        List of cards with status, limits, expiry, and network.
    """
    cards = MOCK_CARDS.get(account_id.upper(), [])
    if not cards:
        return {"error": f"No cards found for account '{account_id}'."}

    if card_type:
        cards = [c for c in cards if c["card_type"].upper() == card_type.upper()]
        if not cards:
            return {"error": f"No {card_type} cards found for account '{account_id}'."}

    return {
        "account_id": account_id.upper(),
        "total_cards": len(cards),
        "cards": cards,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — Fraud Detection
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def detect_fraud(
    account_id: str,
    transaction_amount: float,
    merchant_name: str,
    location: str,
    transaction_time: Optional[str] = None,
) -> dict:
    """
    Run a fraud risk assessment on a proposed or recent transaction.

    Args:
        account_id:          Customer account ID
        transaction_amount:  Amount in INR
        merchant_name:       Merchant or payee name
        location:            City or country of transaction
        transaction_time:    ISO datetime string; defaults to now

    Returns:
        Risk score (0–100), risk level (LOW/MEDIUM/HIGH), flags raised, and recommendation.
    """
    acc = MOCK_ACCOUNTS.get(account_id.upper())
    if not acc:
        return {"error": f"Account '{account_id}' not found."}

    txn_time = transaction_time or datetime.now().isoformat()
    risk_score = 0
    flags = []

    # Rule 1: Large amount
    if transaction_amount > 100000:
        risk_score += 35
        flags.append("LARGE_AMOUNT: Transaction exceeds ₹1,00,000")

    # Rule 2: International transaction (simple heuristic)
    indian_cities = {"mumbai", "delhi", "bangalore", "hyderabad", "chennai",
                     "kolkata", "pune", "jaipur", "ahmedabad", "surat"}
    if location.lower().strip() not in indian_cities:
        risk_score += 30
        flags.append(f"INTERNATIONAL_LOCATION: Transaction from '{location}'")

    # Rule 3: Late night transaction (between 01:00 and 05:00)
    try:
        hour = datetime.fromisoformat(txn_time).hour
        if 1 <= hour <= 5:
            risk_score += 20
            flags.append(f"ODD_HOURS: Transaction at {hour:02d}:xx")
    except ValueError:
        pass

    # Rule 4: Known high-risk merchant keywords
    risky_keywords = ["casino", "lottery", "crypto", "forex", "bet", "gambling"]
    if any(k in merchant_name.lower() for k in risky_keywords):
        risk_score += 40
        flags.append(f"HIGH_RISK_MERCHANT: '{merchant_name}' matches risk keywords")

    # Rule 5: Rapid repeat pattern (mock — check if same merchant in recent txns)
    recent = MOCK_TRANSACTIONS.get(account_id.upper(), [])
    same_merchant_count = sum(1 for t in recent if merchant_name.lower() in t["description"].lower())
    if same_merchant_count >= 3:
        risk_score += 15
        flags.append(f"REPEAT_MERCHANT: '{merchant_name}' seen {same_merchant_count} times recently")

    # Cap score at 100
    risk_score = min(risk_score, 100)

    if risk_score >= 65:
        risk_level = "HIGH"
        recommendation = "BLOCK transaction and send OTP verification to registered mobile."
    elif risk_score >= 35:
        risk_level = "MEDIUM"
        recommendation = "Request additional authentication (OTP / biometric) before processing."
    else:
        risk_level = "LOW"
        recommendation = "Transaction looks normal. Proceed."

    return {
        "account_id":           account_id.upper(),
        "merchant":             merchant_name,
        "amount":               f"₹{transaction_amount:,.2f}",
        "location":             location,
        "risk_score":           risk_score,
        "risk_level":           risk_level,
        "flags_raised":         flags if flags else ["NONE"],
        "recommendation":       recommendation,
        "assessed_at":          datetime.now().strftime("%d %b %Y, %H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 5 — Credit Score
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_credit_score(account_id: str) -> dict:
    """
    Retrieve the latest credit score and credit health report for a customer.

    Args:
        account_id: Customer account ID

    Returns:
        Credit score, rating band, key factors, and last update date.
    """
    acc = MOCK_ACCOUNTS.get(account_id.upper())
    if not acc:
        return {"error": f"Account '{account_id}' not found."}

    # Deterministic mock score from account_id
    seed = int(hashlib.md5(account_id.encode()).hexdigest()[:4], 16)
    score = 650 + (seed % 200)  # Range 650–849

    if score >= 800:
        band, advice = "EXCELLENT", "You qualify for the best loan rates and highest credit limits."
    elif score >= 740:
        band, advice = "VERY GOOD", "You qualify for most financial products at competitive rates."
    elif score >= 670:
        band, advice = "GOOD", "Eligible for most products; paying dues on time will improve further."
    else:
        band, advice = "FAIR", "Consider reducing outstanding balances and paying EMIs on time."

    return {
        "account_id":     account_id.upper(),
        "customer_name":  acc["customer_name"],
        "credit_score":   score,
        "max_score":      900,
        "rating_band":    band,
        "key_factors": {
            "payment_history":      "On-time (96%)",
            "credit_utilisation":   "43%",
            "credit_age":           "6 years 4 months",
            "active_accounts":      3,
            "recent_hard_enquiries": 1,
        },
        "advice":         advice,
        "last_updated":   "10 Jun 2025",
        "bureau":         "CIBIL",
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 6 — Loan Eligibility
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def check_loan_eligibility(
    account_id: str,
    loan_type: str,
    requested_amount: float,
    tenure_months: int = 36,
) -> dict:
    """
    Check loan eligibility and get an EMI estimate.

    Args:
        account_id:        Customer account ID
        loan_type:         One of 'personal', 'home', 'car', 'education', 'gold'
        requested_amount:  Loan amount in INR
        tenure_months:     Repayment period in months

    Returns:
        Eligibility status, approved amount, interest rate, and EMI breakdown.
    """
    acc = MOCK_ACCOUNTS.get(account_id.upper())
    if not acc:
        return {"error": f"Account '{account_id}' not found."}

    lt = loan_type.lower()
    if lt not in MOCK_LOANS:
        return {"error": f"Loan type '{loan_type}' not found. Options: {', '.join(MOCK_LOANS)}"}

    loan = MOCK_LOANS[lt]
    eligible = requested_amount <= loan["max_amount"]
    approved_amount = requested_amount if eligible else loan["max_amount"]
    capped_tenure = min(tenure_months, loan["max_tenure_months"])

    # EMI = P * r * (1+r)^n / ((1+r)^n - 1)
    r = loan["rate"] / 100 / 12
    n = capped_tenure
    if r > 0:
        emi = approved_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)
    else:
        emi = approved_amount / n

    total_payable = emi * n
    total_interest = total_payable - approved_amount

    return {
        "account_id":       account_id.upper(),
        "loan_type":        lt.upper(),
        "eligible":         eligible,
        "requested_amount": f"₹{requested_amount:,.0f}",
        "approved_amount":  f"₹{approved_amount:,.0f}",
        "interest_rate":    f"{loan['rate']}% p.a.",
        "tenure_months":    capped_tenure,
        "monthly_emi":      f"₹{emi:,.0f}",
        "total_interest":   f"₹{total_interest:,.0f}",
        "total_payable":    f"₹{total_payable:,.0f}",
        "note":             None if eligible else f"Requested amount exceeds max ₹{loan['max_amount']:,}. Showing eligibility for max amount.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 7 — Fund Transfer
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def initiate_fund_transfer(
    from_account_id: str,
    to_account_id: str,
    amount: float,
    remarks: Optional[str] = None,
    transfer_mode: str = "IMPS",
) -> dict:
    """
    Initiate a fund transfer between two accounts.

    Args:
        from_account_id: Source account ID
        to_account_id:   Destination account ID
        amount:          Amount in INR (must be > 0)
        remarks:         Optional transfer note
        transfer_mode:   One of 'IMPS', 'NEFT', 'RTGS'

    Returns:
        Transaction reference, status, and timestamp.
    """
    modes = {"IMPS", "NEFT", "RTGS"}
    if transfer_mode.upper() not in modes:
        return {"error": f"Invalid transfer_mode. Use: {', '.join(modes)}"}

    if amount <= 0:
        return {"error": "Transfer amount must be greater than 0."}

    src = MOCK_ACCOUNTS.get(from_account_id.upper())
    if not src:
        return {"error": f"Source account '{from_account_id}' not found."}

    if src["savings_balance"] < amount:
        return {
            "status": "FAILED",
            "reason": f"Insufficient balance. Available: ₹{src['savings_balance']:,.2f}",
        }

    # Mock transaction reference
    ref = "TXN" + str(random.randint(10000, 99999))

    return {
        "status":            "SUCCESS",
        "reference":         ref,
        "from_account":      from_account_id.upper(),
        "to_account":        to_account_id.upper(),
        "amount":            f"₹{amount:,.2f}",
        "transfer_mode":     transfer_mode.upper(),
        "remarks":           remarks or "—",
        "timestamp":         datetime.now().strftime("%d %b %Y, %H:%M:%S"),
        "note":              "This is a simulated transfer. No real funds moved.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 8 — Block / Unblock Card
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def update_card_status(
    account_id: str,
    card_last4: str,
    action: str,
    reason: Optional[str] = None,
) -> dict:
    """
    Block or unblock a specific card.

    Args:
        account_id:  Customer account ID
        card_last4:  Last 4 digits of the card
        action:      'BLOCK' or 'UNBLOCK'
        reason:      Optional reason (e.g. 'Lost card', 'Fraud suspected')

    Returns:
        Updated card status confirmation.
    """
    if action.upper() not in ("BLOCK", "UNBLOCK"):
        return {"error": "action must be 'BLOCK' or 'UNBLOCK'."}

    cards = MOCK_CARDS.get(account_id.upper(), [])
    matched = [c for c in cards if c["card_number_masked"].endswith(card_last4)]

    if not matched:
        return {"error": f"No card ending in '{card_last4}' found on account '{account_id}'."}

    card = matched[0]
    new_status = "BLOCKED" if action.upper() == "BLOCK" else "ACTIVE"

    return {
        "account_id":   account_id.upper(),
        "card":         card["card_number_masked"],
        "card_type":    card["card_type"],
        "network":      card["network"],
        "old_status":   card["status"],
        "new_status":   new_status,
        "action":       action.upper(),
        "reason":       reason or "Customer request",
        "timestamp":    datetime.now().strftime("%d %b %Y, %H:%M:%S"),
        "note":         "Status updated successfully (simulated).",
    }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 9 — Account Summary
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_account_summary(account_id: str) -> dict:
    """
    Return a full 360° summary of a customer's account including balances,
    active cards, credit score, and KYC status.

    Args:
        account_id: Customer account ID

    Returns:
        Comprehensive account overview.
    """
    acc = MOCK_ACCOUNTS.get(account_id.upper())
    if not acc:
        return {"error": f"Account '{account_id}' not found."}

    cards = MOCK_CARDS.get(account_id.upper(), [])
    active_cards = sum(1 for c in cards if c["status"] == "ACTIVE")

    seed = int(hashlib.md5(account_id.encode()).hexdigest()[:4], 16)
    credit_score = 650 + (seed % 200)

    return {
        "account_id":        account_id.upper(),
        "customer_name":     acc["customer_name"],
        "branch":            acc["branch"],
        "ifsc":              acc["ifsc"],
        "account_status":    acc["status"],
        "kyc_status":        acc["kyc"],
        "balances": {
            "savings":  f"₹{acc['savings_balance']:,.2f}",
            "current":  f"₹{acc['current_balance']:,.2f}",
            "fd":        f"₹{acc['fd_balance']:,.2f}",
        },
        "cards": {
            "total":  len(cards),
            "active": active_cards,
        },
        "credit_score":      credit_score,
        "as_of":             datetime.now().strftime("%d %b %Y, %H:%M"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE — Bank Info
# ─────────────────────────────────────────────────────────────────────────────

@mcp.resource("bank://info")
def bank_info() -> str:
    """Static information about NEODesk Bank."""
    return """
    NEODesk Bank — Core Banking System
    ─────────────────────────────────────
    Name:         NEODesk Digital Bank
    Founded:      2020
    Headquarters: Mumbai, India
    IFSC Prefix:  NEOB
    24x7 Support: 1800-NEO-DESK (toll-free)
    Email:        support@neodesk.bank
    MCP Version:  1.0.0

    Supported Transfer Modes:
      - IMPS (Instant, 24x7, up to ₹5L per txn)
      - NEFT (Batch, Mon–Sat)
      - RTGS (Min ₹2L, for large value)

    Supported Loan Types:
      - Personal  · Home  · Car  · Education  · Gold
    """


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT — Fraud investigation helper
# ─────────────────────────────────────────────────────────────────────────────

@mcp.prompt()
def fraud_investigation_prompt(account_id: str, transaction_ref: str) -> str:
    """Generate a prompt for investigating a suspicious transaction."""
    return f"""
    You are a fraud analyst at NEODesk Bank.

    Investigate the following suspicious transaction:
    - Account: {account_id}
    - Transaction Ref: {transaction_ref}

    Steps:
    1. Fetch account summary for {account_id}
    2. Get recent transactions and locate ref {transaction_ref}
    3. Run fraud detection on the transaction details
    4. Check card status
    5. Summarise findings and recommend action (block card, call customer, clear transaction)
    """


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--http" in sys.argv:
        # HTTP/SSE mode — useful for MCP Inspector or browser-based clients
        mcp.run(transport="sse", host="0.0.0.0", port=8000)
    else:
        # Default: stdio mode — for MCP clients like Claude Desktop
        mcp.run(transport="stdio")