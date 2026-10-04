"""Append approved bills to a Google Sheet with 'Bills' and 'Items' tabs."""
from app.config import GOOGLE_CREDS, SHEET_NAME
from app.schemas import Bill


def _cell(value):
    return "" if value is None else value


def save(bill_id: str, filename: str, bill: Bill) -> None:
    import gspread
    sheet = gspread.service_account(filename=GOOGLE_CREDS).open(SHEET_NAME)
    sheet.worksheet("Bills").append_row(
        [bill_id, _cell(bill.date), bill.store_name, bill.category, _cell(bill.payment_method),
         _cell(bill.tax), _cell(bill.total), filename],
        value_input_option="USER_ENTERED")
    if bill.items:
        sheet.worksheet("Items").append_rows(
            [[bill_id, _cell(bill.date), bill.store_name, i.name, _cell(i.qty), _cell(i.price)] for i in bill.items],
            value_input_option="USER_ENTERED")
