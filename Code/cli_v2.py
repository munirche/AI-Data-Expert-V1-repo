"""
Expert Learning System V2 - Command Line Interface

Commands: load, list, stats, reset, search, add, analyze, delete
"""

import argparse
import json
import sys
import os

# Add Code directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expert_learning_system_v2 import ExpertLearningEngine


def cmd_load(engine: ExpertLearningEngine, args):
    """Load records from corpus into database."""

    if args.record:
        # Load specific record(s)
        if '-' in args.record:
            # Range: e.g., "5-10"
            start, end = args.record.split('-')
            records = engine.get_corpus_range(int(start) - 1, int(end))
            count = 0
            for record in records:
                record_id = record[engine.config["record_id_field"]]
                ann_id = engine.load_from_corpus(str(record_id))
                if ann_id:
                    print(f"  Loaded record {record_id} -> annotation {ann_id}")
                    count += 1
            print(f"\n[OK] Loaded {count} records.")
        else:
            # Single record
            ann_id = engine.load_from_corpus(args.record)
            if ann_id:
                print(f"[OK] Loaded record {args.record} -> annotation {ann_id}")
            else:
                print(f"[ERROR] Record {args.record} not found in corpus.")
    else:
        # Load first N records
        n = args.n if args.n else 5
        print(f"Loading first {n} records from corpus...")
        loaded = engine.load_n_from_corpus(n)
        print(f"[OK] Loaded {len(loaded)} records.")


def cmd_list(engine: ExpertLearningEngine, args):
    """List annotations in database."""

    annotations = engine.list_annotations(limit=args.limit, tag=args.tag)

    if not annotations:
        print("No annotations in database.")
        return

    print(f"\n{'ID':<10} {'Record':<8} {'Risk':<10} {'Summary':<50}")
    print("-" * 80)

    for ann in annotations:
        summary = ann['summary'][:47] + "..." if len(ann['summary']) > 50 else ann['summary']
        print(f"{ann['annotation_id']:<10} {ann['record_id']:<8} {ann['risk_assessment']:<10} {summary}")

    print(f"\nTotal: {len(annotations)} annotations")

    if args.full and annotations:
        print("\n" + "=" * 80)
        for ann in annotations:
            full = engine.get_annotation(ann['annotation_id'])
            if full:
                print(f"\n--- Annotation {full['annotation_id']} ---")
                print(f"Record ID: {full['record_id']}")
                print(f"Summary: {full['summary']}")
                print(f"Analysis: {full['analysis']}")
                print(f"Risk: {full['risk_assessment']}")
                print(f"Tags: {', '.join(full['tags'])}")


def cmd_stats(engine: ExpertLearningEngine, args):
    """Show database statistics."""

    stats = engine.get_stats()

    print("\n=== Database Statistics ===\n")
    print(f"Annotations: {stats['total_annotations']}")
    print(f"\nCorpus: {stats['corpus_total']} cases")
    print(f"  Loaded: {stats['corpus_loaded']} ({100*stats['corpus_loaded']//max(1,stats['corpus_total'])}%)")
    print(f"  Remaining: {stats['corpus_remaining']}")

    if stats['risk_distribution']:
        print(f"\nRisk distribution:")
        for risk, count in stats['risk_distribution'].items():
            pct = 100 * count // stats['total_annotations']
            print(f"  {risk}: {count} ({pct}%)")

    if stats['top_tags']:
        print(f"\nTop tags:")
        for tag, count in stats['top_tags']:
            print(f"  {tag}: {count}")


def cmd_reset(engine: ExpertLearningEngine, args):
    """Reset the database."""

    if not args.confirm:
        print("Use --confirm to reset the database.")
        return

    stats = engine.get_stats()
    count = stats['total_annotations']

    if count > 0:
        confirm = input(f"WARNING: This will delete {count} annotations. Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("Cancelled.")
            return

    success = engine.reset_database()

    if success:
        print("[OK] Database cleared.")

        if args.reload:
            print(f"\nReloading {args.reload} records from corpus...")
            loaded = engine.load_n_from_corpus(args.reload)
            print(f"[OK] Loaded {len(loaded)} records.")
    else:
        print("[ERROR] Failed to reset database.")


def cmd_search(engine: ExpertLearningEngine, args):
    """Search for similar annotations."""

    if not args.query:
        print("Usage: cli_v2.py search \"query text\"")
        return

    results = engine.search_similar(args.query, limit=args.limit or 5)

    if not results:
        print("No similar annotations found.")
        return

    print(f"\nSimilar annotations for: \"{args.query}\"\n")

    for i, r in enumerate(results, 1):
        print(f"#{i} ({r['similarity_score']:.0%} match): {r['annotation_id']}")
        print(f"   {r['summary']}")
        if args.full:
            print(f"   Analysis: {r['analysis'][:200]}...")
        print()


def cmd_add(engine: ExpertLearningEngine, args):
    """Add annotation from file."""

    if not args.file:
        print("Usage: cli_v2.py add --file annotation.json")
        return

    try:
        with open(args.file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.file}")
        return
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON: {e}")
        return

    # Validate required fields
    required = ['record_id', 'summary', 'analysis']
    for field in required:
        if field not in data:
            print(f"[ERROR] Missing required field: {field}")
            return

    ann_id = engine.store_annotation(
        record_id=data['record_id'],
        record_data=data.get('record_data', {}),
        summary=data['summary'],
        analysis=data['analysis'],
        risk_assessment=data.get('risk_assessment', ''),
        patterns=data.get('patterns', []),
        recommended_actions=data.get('recommended_actions', []),
        additional_tests=data.get('additional_tests', []),
        tags=data.get('tags', [])
    )

    print(f"[OK] Annotation saved with ID: {ann_id}")


def cmd_delete(engine: ExpertLearningEngine, args):
    """Delete annotation(s) from database."""

    if not args.id and not args.record:
        print("Usage: cli_v2.py delete --id <annotation_id>  or  --record <record_id>")
        return

    if args.id:
        # Show what will be deleted
        ann = engine.get_annotation(args.id)
        if not ann:
            print(f"[ERROR] Annotation {args.id} not found.")
            return

        print(f"Annotation: {ann['annotation_id']}")
        print(f"  Record: {ann['record_id']}")
        print(f"  Summary: {ann['summary'][:60]}...")

        confirm = input("Delete this annotation? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return

        count = engine.delete_annotation(annotation_id=args.id)
        print(f"[OK] Deleted {count} annotation(s).")

    elif args.record:
        # Delete by record_id (may have multiple annotations)
        annotations = engine.list_annotations()
        matches = [a for a in annotations if a['record_id'] == str(args.record)]

        if not matches:
            print(f"[ERROR] No annotations found for record {args.record}.")
            return

        print(f"Found {len(matches)} annotation(s) for record {args.record}:")
        for ann in matches:
            print(f"  - {ann['annotation_id']}: {ann['summary'][:50]}...")

        confirm = input(f"Delete all {len(matches)} annotation(s)? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return

        count = engine.delete_annotation(record_id=args.record)
        print(f"[OK] Deleted {count} annotation(s).")


def cmd_analyze(engine: ExpertLearningEngine, args):
    """Analyze a record using AI."""

    # Get record data
    if args.record:
        record = engine.get_corpus_record(args.record)
        if not record:
            print(f"[ERROR] Record {args.record} not found in corpus.")
            return

        # Extract data fields
        data_fields = engine.config["data_fields"]
        record_id_field = engine.config["record_id_field"]
        record_data = {k: record[k] for k in data_fields if k in record}
        record_data[record_id_field] = record[record_id_field]

        ground_truth = {
            'summary': record.get('summary', ''),
            'analysis': record.get('analysis', ''),
            'risk_assessment': record.get('risk_assessment', '')
        }

    elif args.file:
        try:
            with open(args.file, 'r') as f:
                record_data = json.load(f)
            ground_truth = None
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[ERROR] {e}")
            return
    else:
        print("Usage: cli_v2.py analyze --record ID  or  --file record.json")
        return

    # Retrieve similar
    print("Retrieving similar past analyses...")
    similar = engine.retrieve_similar(record_data, n=3)
    print(f"Found {len(similar)} similar cases.\n")

    # Generate AI analysis
    print("Generating AI analysis...\n")
    result = engine.generate_analysis(record_data, similar)

    print("=" * 70)
    print("RECORD DATA:")
    print("-" * 70)
    for k, v in record_data.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("AI ANALYSIS:")
    print("-" * 70)

    if not result.get('parse_success', False):
        print("[WARNING] AI did not return valid JSON. Showing raw response:")
        print(result.get('raw_response', result.get('analysis', '')))
    else:
        print(f"\nSummary: {result['summary']}")
        print(f"\nRisk Assessment: {result['risk_assessment']}")
        print(f"\nAnalysis:\n{result['analysis']}")

        if result['patterns']:
            print(f"\nPatterns Detected:")
            for p in result['patterns']:
                print(f"  - [{p.get('type', 'unknown')}] {p.get('field', '')}: {p.get('description', '')}")

        if result['recommended_actions']:
            print(f"\nRecommended Actions:")
            for a in result['recommended_actions']:
                print(f"  - {a}")

        if result['additional_tests']:
            print(f"\nAdditional Tests:")
            for t in result['additional_tests']:
                print(f"  - {t}")

    print("=" * 70)

    # Compare mode
    if args.compare and ground_truth:
        print("\nGROUND TRUTH (from corpus):")
        print("-" * 70)
        print(f"Summary: {ground_truth['summary']}")
        print(f"Analysis: {ground_truth['analysis']}")
        print(f"Risk: {ground_truth['risk_assessment']}")
        print("=" * 70)

    # Interactive review (unless batch mode)
    if not args.batch:
        print("\nOptions: [A]ccept  [E]dit  [R]eject  [S]kip")
        choice = input("Choice: ").strip().upper()

        if choice == 'A':
            # Save with all structured fields
            ann_id = engine.store_annotation(
                record_id=str(record_data.get(engine.config["record_id_field"], "unknown")),
                record_data=record_data,
                summary=result.get('summary', ''),
                analysis=result.get('analysis', ''),
                risk_assessment=result.get('risk_assessment', ''),
                patterns=result.get('patterns', []),
                recommended_actions=result.get('recommended_actions', []),
                additional_tests=result.get('additional_tests', []),
                tags=["ai_generated"]
            )
            print(f"[OK] Saved annotation {ann_id}")

        elif choice == 'E':
            # Save to temp JSON file for editing
            temp_file = "temp_analysis.json"
            edit_data = {
                'summary': result.get('summary', ''),
                'analysis': result.get('analysis', ''),
                'risk_assessment': result.get('risk_assessment', ''),
                'patterns': result.get('patterns', []),
                'recommended_actions': result.get('recommended_actions', []),
                'additional_tests': result.get('additional_tests', [])
            }
            with open(temp_file, 'w') as f:
                json.dump(edit_data, f, indent=2)
            print(f"\nEdit the file: {temp_file}")
            input("Press Enter when done editing...")

            with open(temp_file, 'r') as f:
                edited = json.load(f)

            ann_id = engine.store_annotation(
                record_id=str(record_data.get(engine.config["record_id_field"], "unknown")),
                record_data=record_data,
                summary=edited.get('summary', ''),
                analysis=edited.get('analysis', ''),
                risk_assessment=edited.get('risk_assessment', ''),
                patterns=edited.get('patterns', []),
                recommended_actions=edited.get('recommended_actions', []),
                additional_tests=edited.get('additional_tests', []),
                tags=["expert_reviewed"]
            )
            print(f"[OK] Saved annotation {ann_id}")
            os.remove(temp_file)

        elif choice == 'R':
            print("Rejected. Write your own analysis:")
            # Save empty template for writing
            temp_file = "temp_analysis.json"
            template = {
                'summary': '',
                'analysis': '',
                'risk_assessment': 'low',
                'patterns': [],
                'recommended_actions': [],
                'additional_tests': []
            }
            with open(temp_file, 'w') as f:
                json.dump(template, f, indent=2)
            print(f"\nEdit the file: {temp_file}")
            input("Press Enter when done...")

            with open(temp_file, 'r') as f:
                written = json.load(f)

            ann_id = engine.store_annotation(
                record_id=str(record_data.get(engine.config["record_id_field"], "unknown")),
                record_data=record_data,
                summary=written.get('summary', ''),
                analysis=written.get('analysis', ''),
                risk_assessment=written.get('risk_assessment', ''),
                patterns=written.get('patterns', []),
                recommended_actions=written.get('recommended_actions', []),
                additional_tests=written.get('additional_tests', []),
                tags=["expert_written"]
            )
            print(f"[OK] Saved annotation {ann_id}")
            os.remove(temp_file)

        else:
            print("Skipped. Nothing saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Expert Learning System V2 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # load command
    load_parser = subparsers.add_parser('load', help='Load records from corpus')
    load_parser.add_argument('n', type=int, nargs='?', help='Number of records to load')
    load_parser.add_argument('--record', type=str, help='Specific record ID or range (e.g., 5-10)')

    # list command
    list_parser = subparsers.add_parser('list', help='List annotations')
    list_parser.add_argument('--limit', type=int, help='Limit number of results')
    list_parser.add_argument('--tag', type=str, help='Filter by tag')
    list_parser.add_argument('--full', action='store_true', help='Show full details')

    # stats command
    subparsers.add_parser('stats', help='Show database statistics')

    # reset command
    reset_parser = subparsers.add_parser('reset', help='Reset database')
    reset_parser.add_argument('--confirm', action='store_true', help='Confirm reset')
    reset_parser.add_argument('--reload', type=int, help='Reload N records after reset')

    # search command
    search_parser = subparsers.add_parser('search', help='Search similar annotations')
    search_parser.add_argument('query', type=str, nargs='?', help='Search query')
    search_parser.add_argument('--limit', type=int, help='Limit results')
    search_parser.add_argument('--full', action='store_true', help='Show full analysis')

    # add command
    add_parser = subparsers.add_parser('add', help='Add annotation from file')
    add_parser.add_argument('--file', type=str, help='Path to annotation JSON file')

    # delete command
    delete_parser = subparsers.add_parser('delete', help='Delete annotation(s)')
    delete_parser.add_argument('--id', type=str, help='Annotation ID to delete')
    delete_parser.add_argument('--record', type=str, help='Delete all annotations for record ID')

    # analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a record with AI')
    analyze_parser.add_argument('--record', type=str, help='Record ID from corpus')
    analyze_parser.add_argument('--file', type=str, help='Path to record JSON file')
    analyze_parser.add_argument('--compare', action='store_true', help='Compare with ground truth')
    analyze_parser.add_argument('--batch', action='store_true', help='Batch mode (no prompts)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize engine
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    engine = ExpertLearningEngine(config_path=config_path)

    # Dispatch command
    commands = {
        'load': cmd_load,
        'list': cmd_list,
        'stats': cmd_stats,
        'reset': cmd_reset,
        'search': cmd_search,
        'add': cmd_add,
        'delete': cmd_delete,
        'analyze': cmd_analyze
    }

    commands[args.command](engine, args)


if __name__ == "__main__":
    main()
