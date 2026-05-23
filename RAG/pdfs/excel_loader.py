import os
import glob
import pandas as pd
from langchain_core.documents import Document


def excel_to_documents(excel_directory):
    """
    Read all Excel files in the directory and convert each row into a Document.
    """
    all_docs = []

    excel_files = glob.glob(f"{excel_directory}/*.xlsx")

    if not excel_files:
        print("[INFO] No Excel files found.")
        return []

    print(f"[INFO] Found {len(excel_files)} Excel file(s)")

    for excel_file in excel_files:
        file_name = os.path.basename(excel_file)
        print(f"\n[INFO] Reading: {file_name}")

        try:
            all_sheets = pd.read_excel(excel_file, sheet_name=None)

            for sheet_name, df in all_sheets.items():
                print(f"   → Sheet: {sheet_name}")
                print(f"   → Rows : {len(df)}")

                df = df.dropna(how="all")

                for row_idx, row in df.iterrows():
                    content = []
                    for column, value in row.items():
                        if pd.isna(value):
                            continue
                        content.append(f"{column}: {value}")

                    if not content:
                        continue

                    doc = Document(
                        page_content="\n".join(content),
                        metadata={
                            "source": file_name,
                            "sheet": sheet_name,
                            "row": int(row_idx),
                            "type": "excel",
                        },
                    )
                    all_docs.append(doc)

        except Exception as error:
            print(f"[!] Error reading {file_name}: {error}")

    print(f"\n[OK] Total Excel Documents: {len(all_docs)}")
    return all_docs


def csv_to_documents(csv_directory):
    """
    Read all CSV files in the directory and convert each row into a Document.
    """
    all_docs = []

    csv_files = glob.glob(f"{csv_directory}/*.csv")

    if not csv_files:
        print("[INFO] No CSV files found.")
        return []

    print(f"[INFO] Found {len(csv_files)} CSV file(s)")

    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        print(f"\n[INFO] Reading: {file_name}")

        try:
            df = pd.read_csv(csv_file, encoding="utf-8")
            df = df.dropna(how="all")

            print(f"   → Rows : {len(df)}")

            for row_idx, row in df.iterrows():
                content = []
                for column, value in row.items():
                    if pd.isna(value):
                        continue
                    content.append(f"{column}: {value}")

                if not content:
                    continue

                doc = Document(
                    page_content="\n".join(content),
                    metadata={
                        "source": file_name,
                        "row": int(row_idx),
                        "type": "csv",
                    },
                )
                all_docs.append(doc)

        except Exception as error:
            print(f"[!] Error reading {file_name}: {error}")

    print(f"\n[OK] Total CSV Documents: {len(all_docs)}")
    return all_docs
