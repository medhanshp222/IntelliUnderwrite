import os
import re
from datetime import datetime

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
# 1. CONVERT PDF TO IMAGES
# ============================================================

def pdf_to_images(pdf_path):

    pdf = pymupdf.open(pdf_path)

    images = []

    for page in pdf:

        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(3, 3),
            alpha=False
        )

        img = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        images.append(img)

    pdf.close()

    return images


# ============================================================
# 2. PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # Improve OCR resolution
    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    # Remove small noise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Threshold
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

    pages = pdf_to_images(pdf_path)

    all_text = []

    for image in pages:

        processed = preprocess_image(image)

        text = pytesseract.image_to_string(
            processed,
            config="--psm 6"
        )

        all_text.append(text)

    return "\n".join(all_text)


# ============================================================
# 4. AMOUNT CONVERSION
# ============================================================

def amount_to_float(value):

    if not value:
        return 0.0

    value = value.replace(",", "")
    value = value.replace("₹", "")
    value = value.replace("$", "")
    value = value.strip()

    try:
        return float(value)
    except ValueError:
        return 0.0


# ============================================================
# 5. EXTRACT TRANSACTION ROWS
# ============================================================

def extract_transactions(text):

    transactions = []

    lines = text.splitlines()

    # Normal HDFC date
    date_pattern = r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b"

    # Money value
    amount_pattern = r"\b\d[\d,]*\.\d{2}\b"

    for line in lines:

        line = line.strip()

        if not line:
            continue

        date_match = re.search(
            date_pattern,
            line
        )

        if not date_match:
            continue

        amounts = re.findall(
            amount_pattern,
            line
        )

        # A transaction normally has amount + balance
        if len(amounts) < 2:
            continue

        transaction_date = date_match.group(1)

        # Everything after first date
        remaining = line[
            date_match.end():
        ].strip()

        # OCR statements usually contain:
        #
        # narration | transaction date | amount | balance
        #
        # Therefore:
        # second-last amount = transaction amount
        # last amount        = closing balance

        transaction_amount = amount_to_float(
            amounts[-2]
        )

        closing_balance = amount_to_float(
            amounts[-1]
        )

        # Remove amounts from narration
        narration = remaining

        for amount in amounts:
            narration = narration.replace(
                amount,
                " "
            )

        narration = re.sub(
            r"\s+",
            " ",
            narration
        ).strip()

        transactions.append({
            "date": transaction_date,
            "narration": narration,
            "amount": transaction_amount,
            "balance": closing_balance
        })

    return transactions


# ============================================================
# 6. PARSE DATE
# ============================================================

def parse_date(date_string):

    formats = [
        "%d/%m/%y",
        "%d/%m/%Y"
    ]

    for fmt in formats:

        try:
            return datetime.strptime(
                date_string,
                fmt
            )

        except ValueError:
            continue

    return None


# ============================================================
# 7. FIND SALARY TRANSACTIONS
# ============================================================

def find_salary_transactions(transactions):

    salary_transactions = []

    months = (
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC"
    )

    for transaction in transactions:

        narration = transaction[
            "narration"
        ].upper()

        # ----------------------------------------------------
        # Only accept actual monthly salary narration
        #
        # Examples:
        # SALARY JUNE 19
        # SALARY AUG 19
        # SALARY NOV 19
        #
        # Do NOT accept:
        # EARLYSALARY
        # RAZORPAY EARLY SALARY
        # ----------------------------------------------------

        salary_pattern = (
            r"\bSALARY\b.*\b("
            + "|".join(months)
            + r")\b"
        )

        if re.search(
            salary_pattern,
            narration
        ):

            salary_transactions.append(
                transaction
            )

    return salary_transactions


# ============================================================
# 8. FIND EMI TRANSACTIONS
# ============================================================

def find_emi_transactions(transactions):

    emi_transactions = []

    for transaction in transactions:

        narration = transaction[
            "narration"
        ].upper()

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # EMI is NOT every transaction containing DEBIT.
        #
        # Only the required EMI factors:
        #
        # ACH D-
        # BAJAJ FINEMI
        # RP-CASHE
        # LOAN
        #
        # ----------------------------------------------------

        is_ach_debit = (
            "ACH D-" in narration
            or "ACH DEBIT" in narration
        )

        is_bajaj = (
            "BAJAJ FINEMI" in narration
        )

        is_rp_cashe = (
            "RP-CASHE" in narration
            or "RP CASHE" in narration
        )

        is_loan = (
            "LOAN" in narration
            or "LOANTAP" in narration
        )

        # ----------------------------------------------------
        # Exclude:
        #
        # Salary
        # Loan credits
        # Return charges
        # Bounce charges
        # ----------------------------------------------------

        is_salary = (
            "SALARY" in narration
        )

        is_return_charge = (
            "RETURN CHARGES" in narration
            or "DEBIT RETURN" in narration
            or "ECS RETURN" in narration
            or "NACH RETURN" in narration
        )

        # If transaction contains loan but is actually a
        # credit, don't count it as EMI.
        #
        # Example:
        # IMPS-LOANTAP CREDIT PRODUCT
        #

        is_loan_credit = (
            "CREDIT PRODUCT" in narration
            or "LOAN CREDIT" in narration
        )

        if is_salary:
            continue

        if is_return_charge:
            continue

        if is_loan_credit:
            continue

        if (
            is_ach_debit
            or is_bajaj
            or is_rp_cashe
            or is_loan
        ):

            emi_transactions.append(
                transaction
            )

    return emi_transactions


# ============================================================
# 9. FIND BOUNCE TRANSACTIONS
# ============================================================

def find_bounce_transactions(transactions):

    bounce_transactions = []

    for transaction in transactions:

        narration = transaction[
            "narration"
        ].upper()

        # ----------------------------------------------------
        # ONLY the factors specified:
        #
        # ACH DEBIT RETURN CHARGES
        # ECS DEBIT RETURN
        # OVERDUE
        #
        # ----------------------------------------------------

        is_ach_return = (
            "ACH DEBIT RETURN CHARGES"
            in narration
        )

        is_ecs_return = (
            "ECS DEBIT RETURN"
            in narration
        )

        is_overdue = (
            "OVERDUE"
            in narration
        )

        if (
            is_ach_return
            or is_ecs_return
            or is_overdue
        ):

            bounce_transactions.append(
                transaction
            )

    return bounce_transactions


# ============================================================
# 10. CALCULATE SALARY DATE VARIANCE
# ============================================================

def calculate_salary_date_variance(
    salary_transactions
):

    dates = []

    for transaction in salary_transactions:

        parsed = parse_date(
            transaction["date"]
        )

        if parsed:

            dates.append(
                parsed.day
            )

    if len(dates) < 2:
        return 0

    # Difference between earliest and latest
    # salary credit day
    return max(dates) - min(dates)


# ============================================================
# 11. CALCULATE MONTHLY EMI
# ============================================================

def calculate_monthly_emi(
    emi_transactions
):

    if not emi_transactions:
        return 0.0

    monthly_totals = {}

    for transaction in emi_transactions:

        parsed_date = parse_date(
            transaction["date"]
        )

        if not parsed_date:
            continue

        month_key = (
            parsed_date.year,
            parsed_date.month
        )

        amount = transaction[
            "amount"
        ]

        if month_key not in monthly_totals:
            monthly_totals[month_key] = 0.0

        monthly_totals[
            month_key
        ] += amount

    if not monthly_totals:
        return 0.0

    # Average EMI obligation per month
    return (
        sum(monthly_totals.values())
        / len(monthly_totals)
    )


# ============================================================
# 12. CALCULATE AVERAGE MONTHLY BALANCE
# ============================================================

def calculate_average_monthly_balance(
    transactions
):

    monthly_balances = {}

    for transaction in transactions:

        parsed_date = parse_date(
            transaction["date"]
        )

        if not parsed_date:
            continue

        month_key = (
            parsed_date.year,
            parsed_date.month
        )

        balance = transaction[
            "balance"
        ]

        monthly_balances.setdefault(
            month_key,
            []
        )

        monthly_balances[
            month_key
        ].append(balance)

    if not monthly_balances:
        return 0.0

    # --------------------------------------------------------
    # First calculate average daily/transaction balance
    # for each month.
    # --------------------------------------------------------

    monthly_averages = []

    for balances in monthly_balances.values():

        if balances:

            monthly_average = (
                sum(balances)
                / len(balances)
            )

            monthly_averages.append(
                monthly_average
            )

    if not monthly_averages:
        return 0.0

    # Average of monthly averages
    return (
        sum(monthly_averages)
        / len(monthly_averages)
    )


# ============================================================
# 13. EXTRACT REQUIRED METRICS
# ============================================================

def extract_bank_statement_metrics(
    text
):

    # --------------------------------------------------------
    # Extract transactions
    # --------------------------------------------------------

    transactions = extract_transactions(
        text
    )

    # --------------------------------------------------------
    # Salary
    # --------------------------------------------------------

    salary_transactions = (
        find_salary_transactions(
            transactions
        )
    )

    salary_credits_count = len(
        salary_transactions
    )

    salary_date_variance_days = (
        calculate_salary_date_variance(
            salary_transactions
        )
    )

    # --------------------------------------------------------
    # EMI
    # --------------------------------------------------------

    emi_transactions = (
        find_emi_transactions(
            transactions
        )
    )

    total_monthly_emis = (
        calculate_monthly_emi(
            emi_transactions
        )
    )

    # --------------------------------------------------------
    # Average monthly balance
    # --------------------------------------------------------

    average_monthly_balance = (
        calculate_average_monthly_balance(
            transactions
        )
    )

    # --------------------------------------------------------
    # Bounce
    # --------------------------------------------------------

    bounce_transactions = (
        find_bounce_transactions(
            transactions
        )
    )

    bounce_count = len(
        bounce_transactions
    )

    # --------------------------------------------------------
    # RETURN ONLY REQUIRED VALUES
    # --------------------------------------------------------

    return {
        "salary_credits_count":
            salary_credits_count,

        "salary_date_variance_days":
            salary_date_variance_days,

        "total_monthly_emis":
            round(
                total_monthly_emis,
                2
            ),

        "average_monthly_balance":
            round(
                average_monthly_balance,
                2
            ),

        "bounce_count_6_months":
            bounce_count
    }


# ============================================================
# 14. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    text = extract_text(
        PDF_PATH
    )

    # --------------------------------------------------------
    # Extract metrics
    # --------------------------------------------------------

    metrics = (
        extract_bank_statement_metrics(
            text
        )
    )

    # --------------------------------------------------------
    # CLEAN OUTPUT ONLY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BANK STATEMENT METRICS")
    print("=" * 60)

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

    print("=" * 60)