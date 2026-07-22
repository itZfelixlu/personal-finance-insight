"""
Synthetic Chase-style bank statement generator (Route 1: enriched dataset).

Goal: make transaction categorization a REAL supervised-learning task instead of
a lookup table. The two properties that matter:

  1. Shared generic tokens across merchants within a category
     (e.g. many different Dining merchants all contain "COFFEE"/"GRILL"/"KITCHEN").
     This lets a TF-IDF model generalize to merchants it has never seen.

  2. Realistic description noise: store numbers, city/state, POS markers
     ("SQ *", "TST*", "POS DEBIT"), so the exact string is rarely repeated.

It also injects a few deliberately ambiguous merchants (AMAZON, TARGET, COSTCO)
so the confusion matrix has something to show.

Output columns match the original file exactly:
    Details, Posting Date, Description, Amount, Category

Usage:
    python scripts/generate_dataset.py            # writes default output path
    python scripts/generate_dataset.py --n 2500   # target number of rows
"""

import argparse
import csv
import random
from datetime import date, timedelta

SEED = 6140  # reproducible: same seed -> same dataset

# ---------------------------------------------------------------------------
# Merchant pools per category.
#
# Each merchant is a "brand core". Many brand cores intentionally embed a
# generic category token (COFFEE, MARKET, PHARMACY, ...). That shared token is
# what lets the model generalize to an UNSEEN brand that shares the token.
# A few brand-only merchants (STARBUCKS, NIKE) are kept to add difficulty.
# ---------------------------------------------------------------------------
MERCHANTS = {
    "Dining": [
        "STARBUCKS", "BLUE BOTTLE COFFEE", "PHILZ COFFEE", "PEETS COFFEE",
        "VERVE COFFEE", "JOE'S CAFE", "SUNRISE CAFE", "CORNER CAFE",
        "CHIPOTLE", "PANDA EXPRESS", "IN-N-OUT BURGER", "SHAKE SHACK",
        "FIVE GUYS BURGERS", "GOLDEN GATE GRILL", "BAYVIEW GRILL",
        "TONY'S PIZZA", "BLAZE PIZZA", "ROUND TABLE PIZZA", "THAI KITCHEN",
        "SICHUAN KITCHEN", "RAMEN HOUSE", "SUSHI HOUSE", "TAQUERIA EL SOL",
        "DOORDASH", "UBER EATS", "GRUBHUB", "SWEETGREEN SALADS",
    ],
    "Grocery": [
        "SAFEWAY", "TRADER JOES", "WHOLE FOODS MARKET", "99 RANCH MARKET",
        "LUCKY SUPERMARKET", "SPROUTS FARMERS MARKET", "H MART",
        "SMART FINAL FOODS", "GROCERY OUTLET", "FOODS CO", "MOLLIE STONES MARKET",
        "BERKELEY BOWL MARKET", "NIJIYA MARKET", "GREEN VALLEY GROCERY",
    ],
    "Shopping": [
        "AMAZON", "NIKE", "TARGET", "BEST BUY", "APPLE STORE", "MACYS",
        "NORDSTROM", "UNIQLO", "IKEA HOME", "HOME DEPOT", "COSTCO",
        "SEPHORA BEAUTY", "REI OUTDOOR", "LULULEMON", "ETSY SHOP",
        "WAYFAIR HOME", "OLD NAVY", "H AND M",
    ],
    "Entertainment": [
        "AMC THEATRES", "REGAL CINEMAS", "CENTURY THEATRES", "STEAM GAMES",
        "NINTENDO", "PLAYSTATION STORE", "XBOX GAMES", "DAVE AND BUSTERS",
        "TICKETMASTER EVENTS", "STUBHUB TICKETS", "TOPGOLF",
    ],
    "Transport": [
        "UBER TRIP", "LYFT RIDE", "CHEVRON GAS", "SHELL GAS", "ARCO GAS",
        "76 GAS STATION", "CALTRAIN", "BART", "AC TRANSIT", "CLIPPER TRANSIT",
        "PARKING METER", "SPPLUS PARKING", "VALVOLINE OIL",
    ],
    "Utilities": [
        "PG&E PAYMENT", "XFINITY INTERNET", "AT&T WIRELESS", "T-MOBILE WIRELESS",
        "VERIZON WIRELESS", "RECOLOGY WASTE", "EBMUD WATER", "SONIC INTERNET",
    ],
    "Subscription": [
        "NETFLIX", "SPOTIFY", "OPENAI CHATGPT", "YOUTUBE PREMIUM",
        "HULU", "DISNEY PLUS", "ADOBE CREATIVE", "NYTIMES DIGITAL",
        "ICLOUD STORAGE", "GITHUB", "AMAZON PRIME MEMBERSHIP",
    ],
    "Healthcare": [
        "CVS PHARMACY", "WALGREENS PHARMACY", "KAISER PERMANENTE", "RITE AID PHARMACY",
        "ONE MEDICAL CLINIC", "SUTTER HEALTH", "QUEST DIAGNOSTICS LAB",
        "BRIGHT SMILE DENTAL", "BAY AREA DENTAL", "LENSCRAFTERS VISION",
    ],
    "Travel": [
        "UNITED AIRLINES", "DELTA AIR LINES", "SOUTHWEST AIRLINES",
        "ALASKA AIRLINES", "HILTON HOTELS", "MARRIOTT HOTELS", "HYATT HOTELS",
        "AIRBNB", "EXPEDIA TRAVEL", "ENTERPRISE RENT A CAR", "HERTZ RENTAL",
    ],
    "Housing": [
        "GREYSTAR PROPERTY", "AVALON APARTMENTS", "EQUITY RESIDENTIAL RENT",
        "IRVINE COMPANY RENT",
    ],
    "Transfer": [
        "ZELLE TRANSFER", "VENMO TRANSFER", "ONLINE TRANSFER", "CASH APP",
        "WIRE TRANSFER", "PAYPAL TRANSFER",
    ],
}

# Deliberately ambiguous merchants: brand core -> {category: probability}.
# When one of these is picked, its label is drawn from this distribution
# instead of its "home" category, creating a handful of genuinely hard cases.
AMBIGUOUS = {
    "AMAZON": {"Shopping": 0.85, "Subscription": 0.15},
    "TARGET": {"Shopping": 0.65, "Grocery": 0.35},
    "COSTCO": {"Shopping": 0.55, "Grocery": 0.45},
}

# Rough per-year transaction volume per category (drives class imbalance too).
YEARLY_COUNTS = {
    "Dining": 700, "Shopping": 430, "Grocery": 340, "Entertainment": 170,
    "Transport": 250, "Utilities": 60, "Subscription": 130, "Healthcare": 60,
    "Travel": 55, "Housing": 12, "Transfer": 95,
}

# Typical amount range (USD, spending magnitude) per category.
AMOUNT_RANGE = {
    "Dining": (5, 65), "Shopping": (15, 450), "Grocery": (18, 220),
    "Entertainment": (8, 90), "Transport": (2, 85), "Utilities": (30, 210),
    "Subscription": (5, 30), "Healthcare": (15, 500), "Travel": (80, 1300),
    "Housing": (1800, 3400), "Transfer": (25, 2000),
}

# Recurring monthly bills post on a stable-ish day of month.
MONTHLY_RECURRING = {"Housing", "Utilities", "Subscription"}

CITIES = [
    "SAN JOSE CA", "PALO ALTO CA", "SAN FRANCISCO CA", "OAKLAND CA",
    "MOUNTAIN VIEW CA", "BERKELEY CA", "SANTA CLARA CA", "FREMONT CA",
    "SUNNYVALE CA", "DALY CITY CA",
]

# POS / processor prefixes seen on real Chase statements.
PREFIXES = ["", "", "", "SQ *", "TST* ", "POS DEBIT ", "PAYPAL *", "CKCD ", "PY *"]
SUFFIXES = ["", "", "", " INC", " LLC", ".COM", " #{n}", " STORE {n}", " {city}"]


def make_description(brand, rng):
    """Turn a brand core into a noisy, realistic statement line."""
    desc = brand
    if rng.random() < 0.55:
        desc = rng.choice(PREFIXES) + desc
    suffix = rng.choice(SUFFIXES)
    suffix = suffix.replace("{n}", str(rng.randint(100, 99999)))
    suffix = suffix.replace("{city}", rng.choice(CITIES))
    desc = desc + suffix
    if rng.random() < 0.15:
        desc = desc.lower() if rng.random() < 0.5 else desc.upper()
    return " ".join(desc.split()).strip()


def make_amount(category, rng):
    lo, hi = AMOUNT_RANGE[category]
    if category in ("Housing", "Utilities", "Travel"):
        amt = round(rng.uniform(lo, hi), 2)
    else:
        # skew toward the low end (most spends are small)
        amt = round(lo + (hi - lo) * (rng.random() ** 2), 2)
    return amt


def resolve_category(brand, home_category, rng):
    """Apply ambiguity: some brands don't always map to their home category."""
    if brand in AMBIGUOUS:
        dist = AMBIGUOUS[brand]
        r, cum = rng.random(), 0.0
        for cat, p in dist.items():
            cum += p
            if r <= cum:
                return cat
    return home_category


def random_date(rng, year=2025):
    start = date(year, 1, 1)
    return start + timedelta(days=rng.randint(0, 364))


def monthly_dates(rng, year=2025):
    """One date per month on a stable-ish day (for recurring bills)."""
    base_day = rng.randint(1, 5)
    for month in range(1, 13):
        jitter = rng.randint(0, 3)
        day = min(base_day + jitter, 28)
        yield date(year, month, day)


def generate(n_target, seed=SEED):
    rng = random.Random(seed)
    rows = []

    for category, brands in MERCHANTS.items():
        if category in MONTHLY_RECURRING:
            # recurring bills: each active merchant posts ~monthly
            n_active = max(1, round(YEARLY_COUNTS[category] / 12))
            active = rng.sample(brands, min(n_active, len(brands)))
            for brand in active:
                for d in monthly_dates(rng):
                    rows.append(build_row(brand, category, d, rng))
        else:
            count = YEARLY_COUNTS[category]
            for _ in range(count):
                brand = rng.choice(brands)
                d = random_date(rng)
                rows.append(build_row(brand, category, d, rng))

    rng.shuffle(rows)
    rows.sort(key=lambda r: r["_date"])  # chronological, like a real statement

    # scale toward n_target if requested (down-sample discretionary spend)
    if n_target and len(rows) > n_target:
        keep = set(rng.sample(range(len(rows)), n_target))
        rows = [r for i, r in enumerate(rows) if i in keep]

    return rows


def build_row(brand, home_category, d, rng):
    category = resolve_category(brand, home_category, rng)
    amount = make_amount(category, rng)
    is_credit = category == "Transfer" and rng.random() < 0.4
    signed = amount if is_credit else -amount
    return {
        "_date": d,
        "Details": "CREDIT" if is_credit else "DEBIT",
        "Posting Date": d.strftime("%m/%d/%Y"),
        "Description": make_description(brand, rng),
        "Amount": f"{signed:.2f}",
        "Category": category,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0,
                        help="target row count (0 = keep all generated)")
    parser.add_argument("--out", default="inputs/synthetic_chase_statement_2025.csv")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rows = generate(args.n, args.seed)

    fields = ["Details", "Posting Date", "Description", "Amount", "Category"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.out}")
    counts = {}
    for r in rows:
        counts[r["Category"]] = counts.get(r["Category"], 0) + 1
    for cat, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<14} {c}")


if __name__ == "__main__":
    main()
