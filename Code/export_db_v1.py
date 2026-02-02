"""
Export DB - Database Export Utility
Exports LanceDB tables to CSV and JSON formats for easy viewing.
"""

import os
import sys
import json
import argparse
import lancedb


def get_available_databases(search_path: str = ".") -> list:
    """Find all LanceDB database folders (ending with _db)."""
    databases = []
    for item in os.listdir(search_path):
        if item.endswith("_db") and os.path.isdir(os.path.join(search_path, item)):
            databases.append(item)
    return databases


def get_tables(db_path: str) -> list:
    """Get all table names in a database."""
    db = lancedb.connect(db_path)
    return db.table_names()


def export_to_csv(db_path: str, table_name: str, output_path: str) -> int:
    """Export a table to CSV format."""
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    df = table.to_pandas()

    # Drop the vector column (not useful in CSV)
    if "vector" in df.columns:
        df = df.drop(columns=["vector"])

    df.to_csv(output_path, index=False)
    return len(df)


def export_to_json(db_path: str, table_name: str, output_path: str) -> int:
    """Export a table to JSON format."""
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    df = table.to_pandas()

    # Drop the vector column (not useful in JSON)
    if "vector" in df.columns:
        df = df.drop(columns=["vector"])

    records = df.to_dict("records")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return len(records)


def show_database_info(db_path: str):
    """Display information about a database."""
    db = lancedb.connect(db_path)
    tables = db.table_names()

    print(f"\nDatabase: {db_path}")
    print(f"Tables: {len(tables)}")
    print("-" * 40)

    for table_name in tables:
        table = db.open_table(table_name)
        count = len(table)
        print(f"  {table_name}: {count} records")


def main():
    parser = argparse.ArgumentParser(
        description="Export LanceDB tables to CSV or JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_db_v1.py --list                     List available databases
  python export_db_v1.py --info                     Show database details
  python export_db_v1.py --csv                      Export to CSV
  python export_db_v1.py --json                     Export to JSON
  python export_db_v1.py --csv --json               Export to both formats
  python export_db_v1.py --db ./my_db --csv         Export specific database
        """
    )

    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to database folder (auto-detects if not specified)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available databases"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show database information"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export to CSV format"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Export to JSON format"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for exported files (default: current directory)"
    )

    args = parser.parse_args()

    # List available databases
    if args.list:
        databases = get_available_databases()
        if databases:
            print("Available databases:")
            for db in databases:
                print(f"  {db}")
        else:
            print("No databases found (folders ending with _db)")
        return

    # Determine which database to use
    db_path = args.db
    if db_path is None:
        databases = get_available_databases()
        if len(databases) == 0:
            print("Error: No databases found. Use --db to specify path.")
            sys.exit(1)
        elif len(databases) == 1:
            db_path = databases[0]
            print(f"Auto-detected database: {db_path}")
        else:
            print("Multiple databases found. Please specify with --db:")
            for db in databases:
                print(f"  {db}")
            sys.exit(1)

    # Check database exists
    if not os.path.isdir(db_path):
        print(f"Error: Database not found: {db_path}")
        sys.exit(1)

    # Show info
    if args.info:
        show_database_info(db_path)
        return

    # If no export format specified, show info by default
    if not args.csv and not args.json:
        show_database_info(db_path)
        print("\nUse --csv or --json to export data.")
        return

    # Get tables
    tables = get_tables(db_path)
    if not tables:
        print("No tables found in database.")
        return

    # Create output directory if needed
    if args.output_dir != "." and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Export each table
    for table_name in tables:
        if args.csv:
            output_file = os.path.join(args.output_dir, f"{table_name}.csv")
            count = export_to_csv(db_path, table_name, output_file)
            print(f"[OK] Exported {count} records to {output_file}")

        if args.json:
            output_file = os.path.join(args.output_dir, f"{table_name}.json")
            count = export_to_json(db_path, table_name, output_file)
            print(f"[OK] Exported {count} records to {output_file}")


if __name__ == "__main__":
    main()
