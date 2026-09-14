import os
import re
from collections import defaultdict
from datetime import datetime
from statistics import median

import pymupdf
import pytesseract
import cv2
import numpy as np
from PIL import Image


# ============================================================
# PDF PATH
# ============================================================

PDF_PATH = os.path.join(
    os.path.dirname(__file__),
    "salary_slip.pdf"
)


# ============================================================
# 1. PDF -> IMAGES
# ============================================================

def pdf_to_images(pdf_path):

    pdf = pymupdf.open(pdf_path)

    images = []

    for page in pdf:

        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(3, 3),
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        images.append(image)

    pdf.close()

    return images


# ============================================================
# 2. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return threshold


# ============================================================
# 3. OCR EXTRACTION
# ============================================================

def extract_text(pdf_path):

    pages = pdf_to_images(
        pdf_path
    )

    all_text = []

    for image in pages:

        processed = preprocess_image(
            image
        )

        text = pytesseract.image_to_string(
            processed,
            config="--psm 6"
        )

        all_text.append(text)

    return "\n".join(all_text)


# ============================================================
# REGEX
# ============================================================

DATE_PATTERN = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
)

AMOUNT_PATTERN = re.compile(
    r"(?<!\d)\d[\d,]*\.\d{2}(?!\d)"
)


# ============================================================
# DATE PARSER
# ============================================================

def parse_date(value):

    value = value.replace(
        "-",
        "/"
    )

    value = value.replace(
        ".",
        "/"
    )

    for fmt in (
        "%d/%m/%Y",
        "%d/%m/%y"
    ):

        try:

            return datetime.strptime(
                value,
                fmt
            )

        except ValueError:
            continue

    return None


# ============================================================
# AMOUNT PARSER
# ============================================================

def amount_to_float(value):

    value = value.replace(
        ",",
        ""
    )

    value = value.replace(
        "₹",
        ""
    )

    try:

        return float(value)

    except ValueError:

        return 0.0


# ============================================================
# CLEAN NARRATION
# ============================================================

def clean_narration(text):

    text = text.upper()

    text = text.replace(
        "–",
        "-"
    )

    text = text.replace(
        "—",
        "-"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 4. EXTRACT TRANSACTION ROWS
#
# Expected OCR structure:
#
# DATE | NARRATION | DATE | AMOUNT | BALANCE
#
# Example:
#
# 05/08/19 | ACH D- ADITY BIRLA FINANCE ...
# | 05/08/19 19,355.00 2,399.05
#
# Last amount  -> Balance
# Previous     -> Transaction amount
# ============================================================

def extract_transactions(text):

    transactions = []

    for raw_line in text.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        date_match = DATE_PATTERN.search(
            line
        )

        if not date_match:
            continue

        date_string = date_match.group()

        parsed_date = parse_date(
            date_string
        )

        if not parsed_date:
            continue

        amounts = AMOUNT_PATTERN.findall(
            line
        )

        # Need transaction amount + balance
        if len(amounts) < 2:
            continue

        values = [
            amount_to_float(x)
            for x in amounts
        ]

        # Last numeric value = balance
        balance = values[-1]

        # Previous numeric value = transaction amount
        amount = values[-2]

        # Remove dates
        narration = DATE_PATTERN.sub(
            " ",
            line
        )

        # Remove amounts
        narration = AMOUNT_PATTERN.sub(
            " ",
            narration
        )

        narration = clean_narration(
            narration
        )

        transactions.append({

            "date": parsed_date,

            "date_string": date_string,

            "narration": narration,

            "amount": amount,

            "balance": balance,

            "raw": line
        })

    return transactions


# ============================================================
# 5. SALARY DETECTION
# ============================================================

def is_real_salary(transaction):

    narration = transaction[
        "narration"
    ]

    # Must contain SALARY
    if "SALARY" not in narration:
        return False

    # Do not count EarlySalary / salary advance
    excluded_words = [

        "EARLYSALARY",

        "EARLY SALARY",

        "RAZPEARLYSALARY",

        "RAZPEARL YSALARY",

        "FLEXSALARY"
    ]

    for word in excluded_words:

        if word in narration:
            return False

    # Do not count debit salary-like transactions
    if "NEFT DR" in narration:
        return False

    if "DEBIT" in narration:
        return False

    return True


def find_salary_transactions(
    transactions
):

    salaries = []

    for transaction in transactions:

        if is_real_salary(
            transaction
        ):

            salaries.append(
                transaction
            )

    return salaries


# ============================================================
# 6. SALARY CREDIT COUNT
#
# Count ONE genuine salary credit per month.
# ============================================================

def calculate_salary_credits_count(
    salaries
):

    months = set()

    for transaction in salaries:

        date = transaction[
            "date"
        ]

        months.add(
            (
                date.year,
                date.month
            )
        )

    return len(months)


# ============================================================
# 7. SALARY DATE VARIANCE
# ============================================================

def calculate_salary_date_variance(
    salaries
):

    if not salaries:
        return 0

    monthly_salary_days = {}

    for transaction in salaries:

        date = transaction[
            "date"
        ]

        month_key = (
            date.year,
            date.month
        )

        # Keep only one salary date per month
        if month_key not in monthly_salary_days:

            monthly_salary_days[
                month_key
            ] = date.day

    days = list(
        monthly_salary_days.values()
    )

    if len(days) < 2:
        return 0

    return (
        max(days)
        -
        min(days)
    )


# ============================================================
# 8. BOUNCE DETECTION
#
# Do NOT count generic RETURN.
# Only actual negative markers.
# ============================================================

BOUNCE_KEYWORDS = [

    "ACH DEBIT RETURN CHARGES",

    "ACH D- RETURN CHARG",

    "ECS DEBIT RETURN",

    "ECS RETURN",

    "NACH DEBIT RETURN",

    "NACH RETURN",

    "CHEQUE RETURN",

    "CHQ RETURN",

    "CHQ RTN",

    "BOUNCE CHARGE",

    "BOUNCED CHARGE",

    "NSF CHARGE",

    "PENALTY CHARGE"
]


def is_bounce(transaction):

    narration = transaction[
        "narration"
    ]

    return any(
        keyword in narration
        for keyword in BOUNCE_KEYWORDS
    )


def find_bounce_transactions(
    transactions
):

    result = []

    for transaction in transactions:

        if is_bounce(
            transaction
        ):

            result.append(
                transaction
            )

    return result


# ============================================================
# 9. EMI / LOAN DETECTION
#
# IMPORTANT:
# "DEBIT" alone is NOT an EMI.
# ============================================================

EMI_LENDER_KEYWORDS = {

    "BAJAJ": [
        "BAJAJ FINEMI",
        "BAJAJ FINEML",
        "BAJAJ FINANCE"
    ],

    "CASHE": [
        "RP-CASHE",
        "RP CASHE"
    ],

    "ADITYA_BIRLA": [
        "ADITY BIRLA FINANCE",
        "ADITYA BIRLA FINANCE"
    ],

    "INDIABULLS": [
        "INDIABULLS",
        "INDIA BULLS"
    ],

    "IVL": [
        "IVL FINANCE"
    ],

    "VIVIFI": [
        "VIVIFI INDIA FINANCE"
    ],

    "LOANTAP": [
        "LOANTAP",
        "LOANTP"
    ],

    "CASHBEAN": [
        "CASHBEAN"
    ],

    "EMI": [
        "EMI"
    ],

    "LOAN": [
        "LOAN INTEREST",
        "LOAN REPAYMENT",
        "LOAN INSTALLMENT"
    ]
}


def identify_emi_lender(
    narration
):

    narration = narration.upper()

    for lender, keywords in (
        EMI_LENDER_KEYWORDS.items()
    ):

        for keyword in keywords:

            if keyword in narration:

                return lender

    # ACH/NACH/ECS are also recurring debit mechanisms
    if "ACH D-" in narration:
        return "ACH"

    if "ACH DEBIT" in narration:
        return "ACH"

    if "NACH" in narration:
        return "NACH"

    if "ECS" in narration:
        return "ECS"

    return None


def is_emi(transaction):

    narration = transaction[
        "narration"
    ]

    # Salary is not EMI
    if is_real_salary(
        transaction
    ):
        return False

    # Bounce is not EMI
    if is_bounce(
        transaction
    ):
        return False

    # Credit-side loan transaction is not EMI
    if "CREDIT PRODUCT" in narration:
        return False

    if "LOAN CREDIT" in narration:
        return False

    lender = identify_emi_lender(
        narration
    )

    return lender is not None


def find_emi_transactions(
    transactions
):

    result = []

    for transaction in transactions:

        if not is_emi(
            transaction
        ):
            continue

        if transaction[
            "amount"
        ] <= 0:
            continue

        result.append(
            transaction
        )

    return result


# ============================================================
# 10. CALCULATE MONTHLY EMI
#
# For every lender:
#
#   lender -> monthly payments
#
# We find the recurring amount.
# ============================================================

def calculate_monthly_emi(
    emi_transactions
):

    if not emi_transactions:
        return 0.0

    lender_month_amounts = defaultdict(
        lambda: defaultdict(list)
    )

    for transaction in emi_transactions:

        narration = transaction[
            "narration"
        ]

        lender = identify_emi_lender(
            narration
        )

        if lender is None:
            continue

        date = transaction[
            "date"
        ]

        month_key = (
            date.year,
            date.month
        )

        amount = round(
            transaction["amount"],
            2
        )

        lender_month_amounts[
            lender
        ][month_key].append(
            amount
        )

    # --------------------------------------------------------
    # Find recurring monthly amount for each lender.
    # --------------------------------------------------------

    recurring_emi_values = []

    for lender, month_data in (
        lender_month_amounts.items()
    ):

        monthly_values = []

        for month, amounts in (
            month_data.items()
        ):

            if not amounts:
                continue

            # Remove exact duplicate OCR entries
            unique_values = list(
                set(amounts)
            )

            # Median handles a possible OCR anomaly
            monthly_value = median(
                unique_values
            )

            monthly_values.append(
                monthly_value
            )

        # A recurring EMI should be visible in at least
        # two months.
        if len(monthly_values) >= 2:

            lender_emi = median(
                monthly_values
            )

            recurring_emi_values.append(
                lender_emi
            )

    return round(
        sum(recurring_emi_values),
        2
    )


# ============================================================
# 11. DAILY CLOSING BALANCES
# ============================================================

def get_daily_closing_balances(
    transactions
):

    ordered = sorted(
        transactions,
        key=lambda x: x["date"]
    )

    daily_balances = {}

    for transaction in ordered:

        date = transaction[
            "date"
        ].date()

        balance = transaction[
            "balance"
        ]

        daily_balances[
            date
        ] = balance

    return daily_balances


# ============================================================
# 12. AVERAGE MONTHLY BALANCE
#
# Your factor:
#
# "Mean of daily closing balance entries"
#
# Therefore:
#
# 1. Take LAST balance for each day.
# 2. Group those balances by month.
# 3. Calculate the monthly average.
# 4. Average the monthly averages.
#
# We DO NOT carry balances into days having no transaction.
# ============================================================

def calculate_average_monthly_balance(
    transactions
):

    daily_balances = (
        get_daily_closing_balances(
            transactions
        )
    )

    if not daily_balances:
        return 0.0

    monthly_balances = defaultdict(
        list
    )

    for date, balance in (
        daily_balances.items()
    ):

        month_key = (
            date.year,
            date.month
        )

        monthly_balances[
            month_key
        ].append(
            balance
        )

    monthly_averages = []

    for balances in (
        monthly_balances.values()
    ):

        if not balances:
            continue

        monthly_average = (
            sum(balances)
            /
            len(balances)
        )

        monthly_averages.append(
            monthly_average
        )

    if not monthly_averages:
        return 0.0

    return round(
        sum(monthly_averages)
        /
        len(monthly_averages),
        2
    )


# ============================================================
# 13. EXTRACT FIVE REQUIRED METRICS
# ============================================================

def extract_bank_statement_metrics(
    text
):

    # Extract transactions
    transactions = (
        extract_transactions(
            text
        )
    )

    # --------------------------------------------------------
    # SALARY
    # --------------------------------------------------------

    salaries = (
        find_salary_transactions(
            transactions
        )
    )

    salary_count = (
        calculate_salary_credits_count(
            salaries
        )
    )

    salary_variance = (
        calculate_salary_date_variance(
            salaries
        )
    )

    # --------------------------------------------------------
    # EMI
    # --------------------------------------------------------

    emis = (
        find_emi_transactions(
            transactions
        )
    )

    monthly_emi = (
        calculate_monthly_emi(
            emis
        )
    )

    # --------------------------------------------------------
    # BALANCE
    # --------------------------------------------------------

    average_balance = (
        calculate_average_monthly_balance(
            transactions
        )
    )

    # --------------------------------------------------------
    # BOUNCES
    # --------------------------------------------------------

    bounces = (
        find_bounce_transactions(
            transactions
        )
    )

    bounce_count = len(
        bounces
    )

    return {

        "salary_credits_count":
            salary_count,

        "salary_date_variance_days":
            salary_variance,

        "total_monthly_emis":
            monthly_emi,

        "average_monthly_balance":
            average_balance,

        "bounce_count_6_months":
            bounce_count
    }


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

def print_metrics(
    metrics
):

    print()

    print(
        "=" * 60
    )

    print(
        "BANK STATEMENT METRICS"
    )

    print(
        "=" * 60
    )

    print(
        f"Salary Credits Count       : "
        f"{metrics['salary_credits_count']}"
    )

    print(
        f"Salary Date Variance       : "
        f"{metrics['salary_date_variance_days']} days"
    )

    print(
        f"Total Monthly EMIs         : "
        f"₹{metrics['total_monthly_emis']:,.2f}"
    )

    print(
        f"Average Monthly Balance    : "
        f"₹{metrics['average_monthly_balance']:,.2f}"
    )

    print(
        f"Bounce Count (6 Months)    : "
        f"{metrics['bounce_count_6_months']}"
    )

    print(
        "=" * 60
    )

    print()


# ============================================================
# 15. MAIN
# ============================================================

if __name__ == "__main__":

    if not os.path.exists(
        PDF_PATH
    ):

        print(
            "ERROR: PDF not found:"
        )

        print(
            PDF_PATH
        )

        raise SystemExit(1)

    # Run OCR
    text = extract_text(
        PDF_PATH
    )

    # Extract required metrics
    metrics = (
        extract_bank_statement_metrics(
            text
        )
    )

    # Print only the five factors
    print_metrics(
        metrics
    )